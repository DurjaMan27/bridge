from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
import logging
from callback_baseline import baseline_bid_from_arrays
from callback_llm import llm_bid_from_arrays

# Set up logging
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

app = FastAPI(title="Bridge Agent Server")

class BridgeState(BaseModel):
    """Pydantic model for the state sent from JAX/callback."""
    observation: list[float]
    current_player: int
    legal_action_mask: list[bool]
    terminated: bool
    rewards: list[float]
    last_bid: int
    last_bidder: int
    call_x: bool
    call_xx: bool
    dealer: int
    shuffled_players: list[int]
    vul_NS: bool
    vul_EW: bool
    bidding_history: list[int]
    # CRITICAL NEW FIELD: Determines which agent's logic to use
    agent_type: str = Field(..., description="Type of agent: 'baseline' or 'llm'")

class ActionResponse(BaseModel):
    """Response model for the calculated action and probabilities."""
    action: int
    pi_probs: list[float]

@app.post("/make_bid", response_model=ActionResponse)
async def make_bid(state: BridgeState):
    """
    Endpoint that routes the bidding request to the appropriate agent logic.
    """
    
    # Convert list/bool inputs from JSON back into numpy arrays/native types
    # This prepares the data for the pure Python agent logic
    state_args = (
        np.asarray(state.observation, dtype=np.float32),
        int(state.current_player),
        np.asarray(state.legal_action_mask, dtype=bool),
        bool(state.terminated),
        np.asarray(state.rewards, dtype=np.float32),
        int(state.last_bid),
        int(state.last_bidder),
        bool(state.call_x),
        bool(state.call_xx),
        int(state.dealer),
        np.asarray(state.shuffled_players, dtype=np.int32),
        bool(state.vul_NS),
        bool(state.vul_EW),
        np.asarray(state.bidding_history, dtype=np.int32),
    )

    try:
        if state.agent_type == 'baseline':
            # Route to the baseline logic
            action_idx, pi_probs = baseline_bid_from_arrays(*state_args)
            logger.info(f"Baseline Agent bid calculated: Action {action_idx}")
        
        elif state.agent_type == 'llm':
            # Route to the LLM logic (which calls OpenAI API)
            action_idx, pi_probs = llm_bid_from_arrays(*state_args)
            logger.info(f"LLM Agent bid calculated: Action {action_idx}")

        else:
            raise ValueError(f"Unknown agent_type: {state.agent_type}")

        return ActionResponse(
            action=int(action_idx),
            pi_probs=pi_probs.tolist()
        )

    except Exception as e:
        logger.error(f"Error in make_bid for agent type {state.agent_type}: {e}")
        # Return a 'Pass' action as a safe fallback
        return ActionResponse(action=0, pi_probs=np.zeros(38, dtype=np.float32).tolist())