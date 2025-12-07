class AgentMockState:
    """
    A lightweight, Python-native object to hold the necessary state for the 
    Python agent to make a bid, mirroring the JAX state structure.
    This is used for the *local* agent call path (server_url=None).
    """
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




# Will be eliminated soon
class MockState:
    """
    A lightweight, Python-native object to hold the necessary state for the 
    Python agent to make a bid, mirroring the JAX state structure.
    """
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