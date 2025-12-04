import jax
import argparse
import subprocess
import time
import threading
import numpy as np
import os # Import os for environment check
from pgx.bridge_bidding import BridgeBidding
from src.eval_manual import make_simple_duplicate_evaluate
from progress_tracker import _bid_counter, reset_counter
# Import both callback types
from src.callback_baseline import make_callback_baseline_agent
from src.callback_llm import make_callback_llm_agent 

# ... (heartbeat function remains the same)
def heartbeat():
    while True:
        time.sleep(10)
        elapsed = time.time() - _bid_counter["start_time"]
        count = _bid_counter["count"]
        rate = count / elapsed if elapsed > 0 else 0
        print(f"[{elapsed:.0f}s] {count} bids ({rate:.1f} bids/sec)")

def get_agent_callback(agent_type, server_url):
    """Helper function to return the correct agent callback based on type."""
    if agent_type == 'baseline':
        return make_callback_baseline_agent(server_url=server_url)
    elif agent_type == 'llm':
        # The LLM callback function needs the server URL if running in server mode
        return make_callback_llm_agent(server_url=server_url)
    elif agent_type in ['DeepMind', 'FAIR']:
        return agent_type
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_server", action="store_true", help="Start agent_server and route bids via HTTP")
    # --- FIX: Changed default port to 8001 to avoid potential conflicts ---
    parser.add_argument("--server_url", default="http://localhost:8001", help="Agent server URL (must match uvicorn host/port)")
    
    # Arguments for agent selection
    parser.add_argument("--team1_agent", default="baseline", choices=["baseline", "llm", "DeepMind", "FAIR"], 
                        help="Agent type for Team 1 (NS)")
    parser.add_argument("--team2_agent", default="baseline", choices=["baseline", "llm", "DeepMind", "FAIR"], 
                        help="Agent type for Team 2 (EW)")
    
    args = parser.parse_args()

    print("These are the args: ", args)

    eval_env = BridgeBidding("dds_results/test_000.npy")
    # rng = jax.random.PRNGKey(0) 

    server_process = None
    try:
        if args.use_server:
            if args.server_url is not None:
                server_stdout = open("src/outputs/server_stdout.log", "w")
                server_stderr = open("src/outputs/server_stderr.log", "w")
                
                # Get port from server_url, defaulting to 80 if not found
                try:
                    server_port = args.server_url.split(':')[-1]
                    if not server_port.isdigit():
                         server_port = "8001"
                except:
                    server_port = "8001"


                print(f"Starting server at {args.server_url} using `python -m uvicorn`...")
                
                # Use 'python -m uvicorn' for better virtual environment compatibility
                server_process = subprocess.Popen(
                    [
                        "python", "-m", "uvicorn", "agent_server:app", 
                        "--host", "127.0.0.1", 
                        "--port", server_port
                    ],
                    stdout=server_stdout,
                    stderr=server_stderr,
                    # Ensure the current environment PATH is used
                    env=os.environ.copy() 
                )
                
                time.sleep(7)  # Increased sleep to 7s to give the server more time to start
                print("Server startup initiated. Proceeding with evaluation.")
            else:
                 print("Server URL is None. Cannot start server.")


        # --- Determine the agents and their configuration ---

        # 1. Get the agent callbacks (which contain the logic/server routing)
        team1_action = get_agent_callback(args.team1_agent, args.server_url if args.use_server else None)
        team2_action = get_agent_callback(args.team2_agent, args.server_url if args.use_server else None)
        
        # 2. Extract model type string for logging/eval (always the agent type string for these)
        team1_model_type = args.team1_agent
        team2_model_type = args.team2_agent

        # 3. Server URL is only passed if --use_server is true
        team1_server_url = args.server_url if args.use_server else None
        team2_server_url = args.server_url if args.use_server else None


        # --- Start Evaluation ---

        reset_counter()
        # Start heartbeat thread for rate monitoring
        threading.Thread(target=heartbeat, daemon=True).start()

        total_envs, batch_size = 80, 10

        args_for_eval = (
            team1_action,
            team1_model_type,
            team2_action,
            team2_model_type,
            team1_server_url,
            team2_server_url,
        )

        all_imps = []
        all_stderrs = []
        all_winrate = []
        
        # ... (Batch processing loop remains the same)
        for batch_start in range(0, total_envs, batch_size):
            batch_end = min(batch_start + batch_size, total_envs)

            batch_rng = jax.random.PRNGKey(batch_start + 12345)
            print(f"Processing envs {batch_start}-{batch_end}")

            duplicate_evaluate = make_simple_duplicate_evaluate(
                eval_env,
                team1_activation=args_for_eval[0],
                team1_model_type=args_for_eval[1],
                team2_activation=args_for_eval[2],
                team2_model_type=args_for_eval[3],
                num_eval_envs = batch_end - batch_start,
                team1_server_url=args_for_eval[4],
                team2_server_url=args_for_eval[5],
            )

            # JIT compilation occurs here
            duplicate_evaluate = jax.jit(duplicate_evaluate)

            log, tablea_info, tableb_info = duplicate_evaluate(
                team1_params=None,
                team2_params=None,
                rng_key=batch_rng,
            )

            all_imps.append(float(log[0]))
            all_stderrs.append(float(log[1]))
            all_winrate.append(float(log[2]))

        final_imp = np.average(all_imps)
        final_stderr = np.sqrt(np.sum(np.array(all_stderrs)**2)) / len(all_stderrs)
        final_winrate = np.average(all_winrate)

        print("-" * 50)
        print(f"Total bids processed: {_bid_counter['count']}")
        print(f"Final IMP: {final_imp:.2f} (StdErr: {final_stderr:.2f})")
        print(f"Final Win Rate (T1 vs T2): {final_winrate*100:.2f}%")
        print("-" * 50)
        print(tablea_info)
        print(tableb_info)
        print("-" * 50)

    finally:
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            server_process.wait()
            # Clean up file handles
            if 'server_stdout' in locals() and not server_stdout.closed: server_stdout.close()
            if 'server_stderr' in locals() and not server_stderr.closed: server_stderr.close()


if __name__ == "__main__":
    main()