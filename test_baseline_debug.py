import jax
import argparse
import subprocess
import time
import threading
import numpy as np
from pgx.bridge_bidding import BridgeBidding
from src.eval_manual import make_simple_duplicate_evaluate
from progress_tracker import _bid_counter, reset_counter

def heartbeat():
    while True:
        time.sleep(10)
        elapsed = time.time() - _bid_counter["start_time"]
        count = _bid_counter["count"]
        rate = count / elapsed if elapsed > 0 else 0
        print(f"[{elapsed:.0f}s] {count} bids ({rate:.1f} bids/sec)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_server", action="store_true", help="Start agent_server and route baseline bids via HTTP")
    parser.add_argument("--server_url", default="http://localhost:8001", help="Agent server URL")
    args = parser.parse_args()

    print("These are the args: ", args)

    eval_env = BridgeBidding("dds_results/test_000.npy")
    rng = jax.random.PRNGKey(0)

    server_process = None
    try:
        if args.use_server:
            server_stdout = open("src/outputs/server_stdout.log", "w")
            server_stderr = open("src/outputs/server_stderr.log", "w")
            server_process = subprocess.Popen(
                ["python", "agent_server.py"],
                # ["bash", "start_server.sh"],
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                stdout=server_stdout,
                stderr=server_stderr,
            )
            time.sleep(2)

        threading.Thread(target=heartbeat, daemon=True).start()
        reset_counter()

        args = (
            "relu",
            "baseline",
            "relu",
            "baseline",
            (args.server_url if args.use_server else None),
            (args.server_url if args.use_server else None),
            None,
            None
        )
        log = batch_envs(eval_env, total_envs=1000, batch_size=20, args=args)

        print("=" * 50)
        print("EVALUATION RESULTS:")
        print(f"IMP: {float(log[0])} +/- {float(log[1])}")
        print(f"Win rate: {float(log[2])}")

    finally:
        if server_process is not None:
            print("SERVER TERMINATED!")
            server_process.terminate()
            server_process.wait()
            server_stdout.close()
            server_stderr.close()

def batch_envs(eval_env: BridgeBidding, total_envs: int, batch_size: int, args: tuple):

    """
        args = (
            team1_action,
            team1_model_type,
            team2_activation,
            team2_model_type,
            team1_server_url,
            team2_server_url,
        )
    """

    all_imps = []
    all_stderrs = []
    all_winrate = []
    for batch_start in range(0, total_envs, batch_size):
        batch_end = min(batch_start + batch_size, total_envs)

        batch_rng = jax.random.PRNGKey(batch_start + 12345)
        print(f"Processing envs {batch_start}-{batch_end}")

        duplicate_evaluate = make_simple_duplicate_evaluate(
            eval_env,
            team1_activation=args[0],
            team1_model_type=args[1],
            team2_activation=args[2],
            team2_model_type=args[3],
            num_eval_envs = batch_end - batch_start,
            team1_server_url=args[4],
            team2_server_url=args[5],
        )

        duplicate_evaluate = jax.jit(duplicate_evaluate)

        log, tablea_info, tableb_info = duplicate_evaluate(
            team1_params=None,
            team2_params=None,
            rng_key=batch_rng,
        )

        all_imps.append(float(log[0]))
        all_stderrs.append(float(log[1]))
        all_winrate.append(float(log[2]))

    overall_imp = np.mean(all_imps)
    combined_stderr = np.sqrt(np.sum(np.array(all_stderrs)**2) / len(all_stderrs))
    overall_winrate = np.mean(all_winrate)
    return [overall_imp, combined_stderr, overall_winrate]

if __name__ == "__main__":
    main()