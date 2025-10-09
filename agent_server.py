from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from baseline import BaselineAgent
import uvicorn

from src.callback_baseline import baseline_bid_from_arrays

app = FastAPI()

class BridgeState(BaseModel):
    observation: list[bool]
    current_player: int
    legal_action_mask: list[bool]
    terminated: bool
    rewards: list[float]
    _last_bid: int
    _last_bidder: int
    _call_x: bool
    _call_xx: bool
    _dealer: int
    _shuffled_players: list[int]
    _vul_NS: bool
    _vul_EW: bool
    _bidding_history: list[int]

class ActionResponse(BaseModel):
    action: int
    pi_probs: list[float]

@app.post("/make_bid", response_model=ActionResponse)
async def make_bid(state: BridgeState):
    try:
        action_idx, pi_probs = baseline_bid_from_arrays(
            np.asarray(state.observation, dtype=bool),
            int(state.current_player),
            np.asarray(state.legal_action_mask, dtype=bool),
            bool(state.terminated),
            np.asarray(state.rewards, dtype=np.float32),
            int(state._last_bid),
            int(state._last_bidder),
            bool(state._call_x),
            bool(state._call_xx),
            int(state._dealer),
            np.asarray(state._shuffled_players, dtype=np.int32),
            bool(state._vul_NS),
            bool(state._vul_EW),
            np.asarray(state._bidding_history, dtype=np.int32),
        )
        return ActionResponse(action=int(action_idx), pi_probs=pi_probs.tolist())
    except Exception as e:
        print("Server error in /make_bid: ", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)