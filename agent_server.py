from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from baseline import BaselineAgent
import uvicorn

from src.callback_baseline import baseline_bid_from_arrays

app = FastAPI()

class BridgeState(BaseModel):
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

class ActionResponse(BaseModel):
    action: int
    pi_probs: list[float]

@app.post("/make_bid", response_model=ActionResponse)
async def make_bid(state: BridgeState):
    try:
        action_idx, pi_probs = baseline_bid_from_arrays(
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
        return ActionResponse(action=int(action_idx), pi_probs=pi_probs.tolist())
    except Exception as e:
        import traceback
        print("="*60)
        print("Server error in /make_bid:")
        print(traceback.format_exc())
        print("="*60)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8001)