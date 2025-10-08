import jax
import jax.numpy as jnp
import numpy as np
import requests
from baseline import BaselineAgent

def make_callback_baseline_agent(server_url: str = None):

    def baseline_agent_callable(state_dict):
        try:
            if server_url:
                response = requests.post(f"{server_url}/make_bid", json=state_dict)
                response.raise_for_status()
                result = response.json()
                return result["action"], result["pi_probs"]
            else:
                agent = BaselineAgent()

                class MockState:
                    def __init__(self, state_data):
                        self.observation = np.array(state_data.observation)
                        self.current_player = state_data.current_player
                        self.legal_action_mask = np.array(state_data.legal_action_mask)
                        self.terminated = state_data.terminated
                        self.rewards = np.array(state_data.rewards)
                        self._last_bid = state_data._last_bid
                        self._last_bidder = state_data._last_bidder
                        self._call_x = state_data._call_x
                        self._call_xx = state_data._call_xx
                        self._dealer = state_data._dealer
                        self._shuffled_players = np.array(state_data._shuffled_players)
                        self._vul_NS = state_data._vul_NS
                        self._vul_EW = state_data._vul_EW
                        self._bidding_history = []
                
                mock_state = MockState(state_dict)
                action = agent.make_bid(mock_state)

                if isinstance(action, str):
                    action_idx = string_to_action_index(action)
                else:
                    action_idx = int(action)

                pi_probs = [0.0] * 38
                pi_probs[action_idx] = 1.0

                return action_idx, pi_probs
        except Exception as e:
            print(f"Error in baseline agent: {e}")
            return 0, [1.0] + [0.0] * 37

    def agent_fn(state):
        """JAX-compatible wrapper using pure_callback"""
        # Convert JAX arrays to Python values for the callback
        state_dict = {
            "observation": state.observation.tolist(),
            "current_player": int(state.current_player),
            "legal_action_mask": state.legal_action_mask.tolist(),
            "terminated": bool(state.terminated),
            "rewards": state.rewards.tolist(),
            "_last_bid": int(state._last_bid),
            "_last_bidder": int(state._last_bidder),
            "_call_x": bool(state._call_x),
            "_call_xx": bool(state._call_xx),
            "_dealer": int(state._dealer),
            "_shuffled_players": state._shuffled_players.tolist(),
            "_vul_NS": bool(state._vul_NS),
            "_vul_EW": bool(state._vul_EW),
        }

        # Use pure_callback to call your Python function
        action, pi_probs = jax.pure_callback(
            baseline_agent_callable,
            (jnp.int32, jnp.array([], dtype=jnp.float32).shape),  # Output types
            state_dict,
            vectorized=False
        )

        return action, jnp.array(pi_probs, dtype=jnp.float32)

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