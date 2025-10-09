import jax
import jax.numpy as jnp
import numpy as np
import requests
from baseline import BaselineAgent
from progress_tracker import increment_bid_count

_session_pool = {}

def get_session(server_url: str):
    """Get or create a session for the given server URL with connection pooling"""
    if server_url not in _session_pool:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,  # Number of connection pools to cache
            pool_maxsize=100,     # Max connections in each pool
            max_retries=3,        # Retry on connection errors
            pool_block=False      # Don't block when pool is full
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        _session_pool[server_url] = session
    return _session_pool[server_url]

class MockState:
    def __init__(self, obs, curr_player, legal_mask, term, rew,
                    last_b, last_bidder, call_x, call_xx,
                    deal, shuffled, vul_ns, vul_ew, bidding_history):
        self.observation = obs
        self.current_player = curr_player
        self.legal_action_mask = legal_mask
        self.terminated = term
        self.rewards = rew
        self._last_bid = last_b
        self._last_bidder = last_bidder
        self._call_x = call_x
        self._call_xx = call_xx
        self._dealer = deal
        self._shuffled_players = shuffled
        self._vul_NS = vul_ns
        self._vul_EW = vul_ew
        self._bidding_history = bidding_history

def baseline_bid_from_arrays(
    observation,
    current_player,
    legal_action_mask,
    terminated,
    rewards,
    last_bid,
    last_bidder,
    call_x,
    call_xx,
    dealer,
    shuffled_players,
    vul_NS,
    vul_EW,
    bidding_history,
):
    agent = BaselineAgent()

    mock_state = MockState(
        observation, current_player, legal_action_mask, terminated, rewards,
        last_bid, last_bidder, call_x, call_xx, dealer, shuffled_players,
        vul_NS, vul_EW, bidding_history
    )
    action = agent.make_bid(mock_state)

    if isinstance(action, str):
        action_idx = string_to_action_index(action)
    else:
        action_idx = int(action)

    pi_probs = np.zeros(38, dtype=np.float32)
    pi_probs[action_idx] = 1.0
    return int(action_idx), pi_probs

def make_callback_baseline_agent(server_url: str = None):

    def baseline_agent_callable(observation, current_player, legal_action_mask,
                                terminated, rewards, last_bid, last_bidder,
                                call_x, call_xx, dealer, shuffled_players, vul_NS,
                                vul_EW, bidding_history):
        try:
            if server_url:
                # print("Connecting to a server for baseline agent...")

                session = get_session(server_url)

                state_dict = {
                    "observation": observation.tolist(),
                    "current_player": int(current_player),
                    "legal_action_mask": legal_action_mask.tolist(),
                    "terminated": bool(terminated),
                    "rewards": rewards.tolist(),
                    "last_bid": int(last_bid),
                    "last_bidder": int(last_bidder),
                    "call_x": bool(call_x),
                    "call_xx": bool(call_xx),
                    "dealer": int(dealer),
                    "shuffled_players": shuffled_players.tolist(),
                    "vul_NS": bool(vul_NS),
                    "vul_EW": bool(vul_EW),
                    "bidding_history": bidding_history.tolist(),
                }

                # response = requests.post(f"{server_url}/make_bid", json=state_dict)
                # response.raise_for_status()
                # result = response.json()

                response = session.post(f"{server_url}/make_bid", json=state_dict)
                result = response.json()

                action = np.int32(result["action"])
                pi_probs = np.array(result["pi_probs"], dtype=np.float32)

                increment_bid_count()
                return action, pi_probs
            else:
                action_idx, pi_probs = baseline_bid_from_arrays(
                    observation, current_player, legal_action_mask,
                    terminated, rewards, last_bid, last_bidder,
                    call_x, call_xx, dealer, shuffled_players, vul_NS,
                    vul_EW, bidding_history
                )

                increment_bid_count()
                return np.int32(action_idx), pi_probs
        except Exception as e:
            import traceback
            print(f"Error in baseline agent: {e}")
            print(traceback.format_exc())

            # return numpy arrays for error case (automatic "Pass")
            pi_probs = np.zeros(38, dtype=np.float32)
            pi_probs[0] = 1.0
            return np.int32(0), pi_probs

    def agent_fn(state):
        """JAX-compatible wrapper using pure_callback"""

        action_shape = jax.ShapeDtypeStruct((), jnp.int32)
        pi_probs_shape = jax.ShapeDtypeStruct((38,), jnp.float32)
        # (jnp.array(0, dtype=jnp.int32), jnp.array([0.0] * 38, dtype=jnp.float32)),

        action, pi_probs = jax.pure_callback(
            baseline_agent_callable,
            (action_shape, pi_probs_shape),
            state.observation,
            state.current_player,
            state.legal_action_mask,
            state.terminated,
            state.rewards,
            state._last_bid,
            state._last_bidder,
            state._call_x,
            state._call_xx,
            state._dealer,
            state._shuffled_players,
            state._vul_NS,
            state._vul_EW,
            state._bidding_history,
            vectorized=False
        )

        # return action, jnp.array(pi_probs, dtype=jnp.float32)
        return action, pi_probs

    return agent_fn


def string_to_action_index(action_str: str) -> int:
    ACTION_IDENTIFIER = {
        0: "Pass", 1: "Double", 2: "Redouble",
        3: "1C", 4: "1D", 5: "1H", 6: "1S", 7: "1NT",
        8: "2C", 9: "2D", 10: "2H", 11: "2S", 12: "2NT",
        13: "3C", 14: "3D", 15: "3H", 16: "3S", 17: "3NT",
        18: "4C", 19: "4D", 20: "4H", 21: "4S", 22: "4NT",
        23: "5C", 24: "5D", 25: "5H", 26: "5S", 27: "5NT",
        28: "6C", 29: "6D", 30: "6H", 31: "6S", 32: "6NT",
        33: "7C", 34: "7D", 35: "7H", 36: "7S", 37: "7NT",
    }

    STRING_TO_ACTION = {v: k for k, v in ACTION_IDENTIFIER.items()}
    return STRING_TO_ACTION.get(action_str, 0)