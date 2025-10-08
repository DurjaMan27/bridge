from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from baseline import BaselineAgent
import uvicorn

app = FastAPI()

class BridgeState(BaseModel):
    observation: list[float]
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

class ActionResponse(BaseModel):
    action: int
    pi_probs: list[float]

@app.post("/make_bid", response_model=ActionResponse)
async def make_bid(state: BridgeState):
    try:
        # Convert to state-like object that Baseline Agent can use
        mock_state = create_mock_state(state)

        # Create agent and get action
        agent = BaselineAgent()
        action = agent.make_bid(mock_state)

        # convert action to proper format
        if isinstance(action, str):
            action_idx = string_to_action_index(action)
        else:
            action_idx = int(action)

        # create probability distribution (deterministic for baseline)
        pi_probs = [0.0] * 38
        pi_probs[action_idx] = 1.0

        return ActionResponse(action = action_idx, pi_probs = pi_probs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_mock_state(state: BridgeState):
    """Create a mocks tate object that BaselineAgent.make_bid() can use"""
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

    return MockState(state)

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