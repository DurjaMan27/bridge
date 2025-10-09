import jax
import argparse
import subprocess
import time
import numpy as np
from pgx.bridge_bidding import BridgeBidding
from src.eval_manual import make_simple_duplicate_evaluate

def decode_state_for_baseline(state):

    ACTION_TO_STRING = {
        0: "Pass",
        1: "Double",
        2: "Redouble",
    }

    BID_LEVELS = ['1', '2', '3', '4', '5', '6', '7']
    BID_SUITS = ['C', 'D', 'H', 'S', 'NT']
    bid_index = 3
    for level in BID_LEVELS:
        for suit in BID_SUITS:
            ACTION_TO_STRING[bid_index] = level + suit
            bid_index += 1

    STRING_TO_ACTION = {v: k for k, v in ACTION_TO_STRING.items()}

    concrete_state = jax.tree_map(lambda x: x[0], state)

    obs = concrete_state.observation
    legal_mask = np.array(concrete_state.legal_action_mask)

    print("\n" + "="*50)
    print("PGX STATE DEBUGGING INFORMATION")
    print("="*50)
    print(f"Current player: {concrete_state.current_player}")
    print(f"Observation shape: {obs.shape}")
    print(f"Terminated: {concrete_state.terminated}")
    print(f"Rewards: {concrete_state.rewards}")
    print(f"Last bid: {concrete_state._last_bid}")
    print(f"Last bidder: {concrete_state._last_bidder}")
    print(f"Call X: {concrete_state._call_x}")
    print(f"Call XX: {concrete_state._call_xx}")
    print(f"Dealer: {concrete_state._dealer}")
    print(f"Shuffled players: {concrete_state._shuffled_players}")
    print(f"Vul NS: {concrete_state._vul_NS}")
    print(f"Vul EW: {concrete_state._vul_EW}")

    print(f"\n=== OBSERVATION BREAKDOWN ===")

    # Vulnerability (obs[0:4])
    vul = obs[0:4]
    print(f"Vulnerability (obs[0:4]): {vul}")
    print(f"  NS: {vul[0]}, EW: {vul[1]}, Both: {vul[2]}, Neither: {vul[3]}")

    # Per player, did this player pass before the opening bid? (obs[4:8])
    passed_before_opening = obs[4:8]
    print(f"Passed before opening (obs[4:8]): {passed_before_opening}")
    for i, passed in enumerate(passed_before_opening):
        print(f"  Player {i}: {'Passed' if passed else 'Did not pass'}")

    # Bidding history against different contracts (obs[8:428])
    print(f"\nBidding history (obs[8:428]):")
    print(f"  Against 1C (obs[8:20]): {obs[8:20]}")
    print(f"  Against 1D (obs[20:32]): {obs[20:32]}")
    print(f"  Against 1H (obs[32:44]): {obs[32:44]}")
    print(f"  Against 1S (obs[44:56]): {obs[44:56]}")
    print(f"  Against 1NT (obs[56:68]): {obs[56:68]}")
    print(f"  Against 2C (obs[68:80]): {obs[68:80]}")
    print(f"  Against 2D (obs[80:92]): {obs[80:92]}")
    print(f"  Against 2H (obs[92:104]): {obs[92:104]}")
    print(f"  Against 2S (obs[104:116]): {obs[104:116]}")
    print(f"  Against 2NT (obs[116:128]): {obs[116:128]}")
    print(f"  Against 3C (obs[128:140]): {obs[128:140]}")
    print(f"  Against 3D (obs[140:152]): {obs[140:152]}")
    print(f"  Against 3H (obs[152:164]): {obs[152:164]}")
    print(f"  Against 3S (obs[164:176]): {obs[176:188]}")
    print(f"  Against 3NT (obs[176:188]): {obs[176:188]}")
    print(f"  Against 4C (obs[188:200]): {obs[188:200]}")
    print(f"  Against 4D (obs[200:212]): {obs[200:212]}")
    print(f"  Against 4H (obs[212:224]): {obs[212:224]}")
    print(f"  Against 4S (obs[224:236]): {obs[224:236]}")
    print(f"  Against 4NT (obs[236:248]): {obs[236:248]}")
    print(f"  Against 5C (obs[248:260]): {obs[248:260]}")
    print(f"  Against 5D (obs[260:272]): {obs[260:272]}")
    print(f"  Against 5H (obs[272:284]): {obs[272:284]}")
    print(f"  Against 5S (obs[284:296]): {obs[284:296]}")
    print(f"  Against 5NT (obs[296:308]): {obs[296:308]}")
    print(f"  Against 6C (obs[308:320]): {obs[308:320]}")
    print(f"  Against 6D (obs[320:332]): {obs[320:332]}")
    print(f"  Against 6H (obs[332:344]): {obs[332:344]}")
    print(f"  Against 6S (obs[344:356]): {obs[344:356]}")
    print(f"  Against 6NT (obs[356:368]): {obs[356:368]}")
    print(f"  Against 7C (obs[368:380]): {obs[368:380]}")
    print(f"  Against 7D (obs[380:392]): {obs[380:392]}")
    print(f"  Against 7H (obs[392:404]): {obs[392:404]}")
    print(f"  Against 7S (obs[404:416]): {obs[404:416]}")
    print(f"  Against 7NT (obs[416:428]): {obs[416:428]}")

    # Cards we hold (obs[428:480]) - 13-hot vector
    hand_cards = obs[428:480]
    print(f"\nHand cards (obs[428:480]): {hand_cards}")
    print(f"  Hand cards as integers: {hand_cards.astype(int)}")

    # Decode the hand cards
    suits = ['C', 'D', 'H', 'S']
    values = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    hand = []
    for i in range(52):
        if hand_cards[i]:
            suit = suits[i // 13]
            value = values[i % 13]
            hand.append(f"{value}{suit}")
    print(f"  Decoded hand: {hand}")

    # Legal actions
    legal_actions = [ACTION_TO_STRING[i] for i, legal in enumerate(legal_mask) if legal]
    print(f"\nLegal actions: {legal_actions}")
    
    return concrete_state

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
            server_process = subprocess.Popen(
                ["python", "agent_server.py"],
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
            )
            time.sleep(2)  # give server time to boot

        duplicate_evaluate = make_simple_duplicate_evaluate(
            eval_env,
            team1_activation="relu",
            team1_model_type="baseline",
            team2_activation="relu",
            team2_model_type="baseline",
            num_eval_envs=1,
            team1_server_url=(args.server_url if args.use_server else None),
            team2_server_url=(args.server_url if args.use_server else None),
        )

        duplicate_evaluate = jax.jit(duplicate_evaluate)

        print("Running baseline vs. baseline evaluation")
        print("This will print detailed state information for debugging.")
        print("="*50)
        log, tablea_info, tableb_info = duplicate_evaluate(
            team1_params=None,
            team2_params=None,
            rng_key=rng
        )

        print("=" * 50)
        print("EVALUATION RESULTS:")
        print(f"IMP: {float(log[0])} +/- {float(log[1])}")
        print(f"Win rate: {float(log[2])}")

    finally:
        if server_process is not None:
            print("SERVER TERMINATED!")
            server_process.terminate()

if __name__ == "__main__":
    main()