import jax
import jax.numpy as jnp
import numpy as np
import requests
import threading  # Added for Semaphore
from src.agents.llm import LLMAgent, llm_bid_from_arrays
from src.utils.state import AgentMockState
from src.utils.progress_tracker import increment_bid_count 

# 1. Global Semaphore to prevent overwhelming the server/API
# Start with a conservative number like 10-20. 
# This ensures only 20 LLM calls are "in-flight" at once.
MAX_CONCURRENT_BIDS = 20
_api_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_BIDS)

_session_pool = {}

def get_session(server_url: str):
    if server_url not in _session_pool:
        session = requests.Session()
        # Reduce pool_maxsize to something more reasonable now that we use a semaphore
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=MAX_CONCURRENT_BIDS, 
            max_retries=3,
            pool_block=True # Block if the pool is full
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        _session_pool[server_url] = session
    return _session_pool[server_url]

def llm_agent_callable(
    observation, current_player, legal_action_mask, terminated, rewards,
    last_bid, last_bidder, call_x, call_xx, dealer, shuffled_players,
    vul_NS, vul_EW, bidding_history, server_url=None
) -> tuple[int, np.ndarray]:
    """
    The main callable function that either makes an HTTP request to the server 
    or calls the LLM Agent locally.
    """
    increment_bid_count()

    if server_url:
        # SERVER PATH: Route through the external FastAPI server
        data = {
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
            "agent_type": "llm" # CRITICAL: Identify the agent type for the server
        }
        
        session = get_session(server_url)
        # response = session.post(f"{server_url}/make_bid", json=data, timeout=300.0) # Increased timeout for LLM
        # response.raise_for_status()
        
        # response_data = response.json()
        # action_idx = np.int32(response_data['action'])
        # pi_probs = np.asarray(response_data['pi_probs'], dtype=np.float32)
        with _api_semaphore:
            try:
                # Explicitly setting (connect_timeout, read_timeout)
                # If you still see 30.0 in the error, the SERVER is timing out.
                response = session.post(
                    f"{server_url}/make_bid", 
                    json=data, 
                    timeout=(10.0, 300.0) 
                )
                response.raise_for_status()
            except requests.exceptions.ReadTimeout:
                print(f"CRITICAL: Server at {server_url} failed to respond within 300s")
                raise

        response_data = response.json()
        action_idx = np.int32(response_data['action'])
        pi_probs = np.asarray(response_data['pi_probs'], dtype=np.float32)
        
    else:
        # LOCAL PATH: Call the LLM Agent directly
        state = AgentMockState(
            observation, current_player, legal_action_mask, terminated, rewards,
            last_bid, last_bidder, call_x, call_xx, dealer, shuffled_players,
            vul_NS, vul_EW, bidding_history
        )
        agent = LLMAgent()
        action_idx, pi_probs = agent.make_bid(state)

    return np.int32(action_idx), pi_probs

def make_callback_llm_agent(server_url=None):
    """
    Returns a JAX-compatible agent function that uses jax.pure_callback
    to call the Python-native LLM agent logic.
    """
    
    def llm_agent_callable_wrapped(obs, curr_player, legal_mask, term, rew, lb, lbr, cx, cxx, deal, shuff, vns, vew, bh):
        return llm_agent_callable(
            obs, curr_player, legal_mask, term, rew, lb, lbr, cx, cxx, deal, shuff, vns, vew, bh, server_url
        )

    def agent_fn(state):
        """JAX-compatible wrapper using pure_callback"""

        # Define the shapes and dtypes of the output for JAX
        action_shape = jax.ShapeDtypeStruct((), jnp.int32)
        pi_probs_shape = jax.ShapeDtypeStruct((38,), jnp.float32)

        action, pi_probs = jax.pure_callback(
            llm_agent_callable_wrapped,
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

        return action, pi_probs

    return agent_fn