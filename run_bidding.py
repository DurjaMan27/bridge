import jax
import argparse
import subprocess
import time
import threading
import numpy as np
import os # Import os for environment check
from pgx.bridge_bidding import BridgeBidding
from src.eval_manual import make_simple_duplicate_evaluate
from src.utils.progress_tracker import _bid_counter, reset_counter
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
    
# --- ADD THIS HELPER TO THE TOP OF YOUR FILE ---
def decode_action(action_idx):
    """Maps integer indices back to bridge bid strings."""
    mapping = {
        0: "Pass", 1: "Double", 2: "Redouble",
        3: "1C", 4: "1D", 5: "1H", 6: "1S", 7: "1NT",
        8: "2C", 9: "2D", 10: "2H", 11: "2S", 12: "2NT",
        13: "3C", 14: "3D", 15: "3H", 16: "3S", 17: "3NT",
        18: "4C", 19: "4D", 20: "4H", 21: "4S", 22: "4NT",
        23: "5C", 24: "5D", 25: "5H", 26: "5S", 27: "5NT",
        28: "6C", 29: "6D", 30: "6H", 31: "6S", 32: "6NT",
        33: "7C", 34: "7D", 35: "7H", 36: "7S", 37: "7NT"
    }
    return mapping.get(int(action_idx), "??")

def inspect_outliers(table_info, batch_start):
    """Prints details of any environment with suspicious scores."""
    # Convert JAX arrays to numpy for easier iteration
    rewards = np.array(table_info.rewards)
    bidding_histories = np.array(table_info.bidding_history)
    last_bids = np.array(table_info.last_bid)
    
    for i in range(rewards.shape[0]):
        # Check if North's score is an outlier (adjust 1500 threshold as needed)
        if abs(rewards[i, 0]) > 1500:

            contract_idx = last_bids[i] + 3 if last_bids[i] != -1 else -1
            contract_name = decode_action(contract_idx)

            print(f"\n[ALERT] Outlier detected at Env {batch_start + i}")
            print(f"Final Contract: {contract_name} (X:{table_info.call_x[i]})")
            print(f"Rewards [N, S, E, W]: {rewards[i]}")
            
            # Decode the auction
            history = [decode_action(bid) for bid in bidding_histories[i] if bid != -1]
            print(f"Auction: {' -> '.join(history)}")
            
            # Identify the final contract
            contract = decode_action(table_info.last_bid[i])
            was_doubled = " (X)" if table_info.call_x[i] else ""
            was_redoubled = " (XX)" if table_info.call_xx[i] else ""
            print(f"Final Contract: {contract}{was_doubled}{was_redoubled} by Player {table_info.last_bidder[i]}")
            print("-" * 30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_server", action="store_true", help="Start agent_server and route bids via HTTP")
    parser.add_argument("--server_url", default="http://localhost:8001", help="Agent server URL (must match uvicorn host/port)")
    
    # Arguments for agent selection
    parser.add_argument("--team1_agent", default="baseline", choices=["baseline", "llm", "DeepMind", "FAIR"], 
                        help="Agent type for Team 1 (NS)")
    parser.add_argument("--team2_agent", default="baseline", choices=["baseline", "llm", "DeepMind", "FAIR"], 
                        help="Agent type for Team 2 (EW)")
    
    args = parser.parse_args()

    print("These are the args: ", args)

    eval_env = BridgeBidding("data/dds_results/test_000.npy")
    # rng = jax.random.PRNGKey(0) 

    server_process = None
    try:
        if args.use_server:
            if args.server_url is not None:
                server_stdout = open("src/logs/server_stdout.log", "w")
                server_stderr = open("src/logs/server_stderr.log", "w")
                
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
                        "python", "-m", "uvicorn", "src.server.agent_server:app", 
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

        total_envs, batch_size = 20, 5
        # 1024, 64

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

            # --- ADD THIS CALL HERE ---
            print(f"Checking for outliers in batch {batch_start}...")
            inspect_outliers(tablea_info, batch_start)
            # --------------------------

            all_imps.append(float(log[0]))
            all_stderrs.append(float(log[1]))
            all_winrate.append(float(log[2]))

        final_imp = np.average(all_imps)
        final_stderr = np.sqrt(np.sum(np.array(all_stderrs)**2)) / len(all_stderrs)
        final_winrate = np.average(all_winrate)

        reverse_imps = []
        reverse_stderrs = []
        reverse_winrate = []

        # ... (Batch processing loop remains the same)
        for batch_start in range(0, total_envs, batch_size):
            batch_end = min(batch_start + batch_size, total_envs)

            batch_rng = jax.random.PRNGKey(batch_start + 12345)
            print(f"Processing envs {batch_start}-{batch_end}")

            duplicate_evaluate = make_simple_duplicate_evaluate(
                eval_env,
                team1_activation=args_for_eval[2],
                team1_model_type=args_for_eval[3],
                team2_activation=args_for_eval[0],
                team2_model_type=args_for_eval[1],
                num_eval_envs = batch_end - batch_start,
                team1_server_url=args_for_eval[5],
                team2_server_url=args_for_eval[4]
            )

            # JIT compilation occurs here
            duplicate_evaluate = jax.jit(duplicate_evaluate)

            reverse_log, reverse_tablea_info, reverse_tableb_info = duplicate_evaluate(
                team1_params=None,
                team2_params=None,
                rng_key=batch_rng,
            )

            reverse_imps.append(float(reverse_log[0]))
            reverse_stderrs.append(float(reverse_log[1]))
            reverse_winrate.append(float(reverse_log[2]))

        reverse_final_imp = np.average(reverse_imps)
        reverse_final_stderr = np.sqrt(np.sum(np.array(reverse_stderrs)**2)) / len(reverse_stderrs)
        reverse_final_winrate = np.average(reverse_winrate)

        print("-" * 50)
        print(f"Total bids processed: {_bid_counter['count']}")
        print(f"Final IMP: {final_imp:.2f} (StdErr: {final_stderr:.2f})")
        print(f"Final Win Rate (T1 vs T2): {final_winrate*100:.2f}%")
        print("-" * 25)
        print(f"Final IMP REVERSED: {reverse_final_imp:.2f} (StdErr: {reverse_final_stderr:.2f})")
        print(f"Final Win Rate (T2 vs T1 - REVERSED): {reverse_final_winrate*100:.2f}%")
        print("-" * 50)
        print(tablea_info)
        print(tableb_info)
        print("-" * 50)
        print(reverse_tablea_info)
        print(reverse_tableb_info)
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