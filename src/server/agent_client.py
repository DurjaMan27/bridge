import jax
import jax.numpy as jnp
import requests
from typing import Callable

def make_http_agent_client(server_url: str = "http://localhost:8000"):
    """Create a JAX-compatible agent that makes HTTP calls"""

    def agent_fn(state):
        # convert JAX arrays to Python lists for JSON serialization
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

        try:
            response = requests.post(f"{server_url}/make_bid", json=state_dict)
            response.raise_for_status()
            result = response.json()

            action = jnp.array(result["action"], dtype=jnp.int32)
            pi_probs = jnp.array(result["pi_probs"], dtype=jnp.float32)

            return action, pi_probs
        except requests.RequestException as e:
            # Fallback to Pass if server is unavailable
            print(f"Server error: {e}")
            return jnp.array(0, dtype=jnp.int32), jnp.zeros(38, dtype=jnp.float32).at[0].set(1.0)

    return agent_fn