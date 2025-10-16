import jax
import jax.numpy as jnp
import numpy as np
import logging
from dotenv import load_dotenv

load_dotenv()

# use o4-mini

class LLMAgent():
  
    _ACTION_IDENTIFIER = {
        0: "Pass", 1: "Double", 2: "Redouble",
        3: "1C", 4: "1D", 5: "1H", 6: "1S", 7: "1NT",
        8: "2C", 9: "2D", 10: "2H", 11: "2S", 12: "2NT",
        13: "3C", 14: "3D", 15: "3H", 16: "3S", 17: "3NT",
        18: "4C", 19: "4D", 20: "4H", 21: "4S", 22: "4NT",
        23: "5C", 24: "5D", 25: "5H", 26: "5S", 27: "5NT",
        28: "6C", 29: "6D", 30: "6H", 31: "6S", 32: "6NT",
        33: "7C", 34: "7D", 35: "7H", 36: "7S", 37: "7NT",
    }

    _CARD_INDEX = {
        0: "CA", 1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6", 6: "C7", 7: "C8", 8: "C9", 9: "C10", 10: "CJ", 11: "CQ", 12: "CK",
        13: "DA", 14: "D2", 15: "D3", 16: "D4", 17: "D5", 18: "D6", 19: "D7", 20: "D8", 21: "D9", 22: "D10", 23: "DJ", 24: "DQ", 25: "DK",
        26: "HA", 27: "H2", 28: "H3", 29: "H4", 30: "H5", 31: "H6", 32: "H7", 33: "H8", 34: "H9", 35: "H10", 36: "HJ", 37: "HQ", 38: "HK",
        39: "SA", 40: "S2", 41: "S3", 42: "S4", 43: "S5", 44: "S6", 45: "S7", 46: "S8", 47: "S9", 48: "S10", 49: "SJ", 50: "SQ", 51: "SK",
    }

    _SUITS = [
        "C", "D", "H", "S"
    ]


    def calc_hcp(hand_cards):
        hcp = 0
        suit_dict = {"C": 0, "D": 0, "H": 0, "S": 0}

        for card_index in hand_cards:
            card_index_str = LLMAgent._CARD_INDEX[card_index]
            suit, card = card_index_str[0], card_index_str[1:]
            suit_dict[suit] += 1

            if card == "A":
                hcp += 4
            elif card == "K":
                hcp += 3
            elif card == "Q":
                hcp += 2
            elif card == "J":
                hcp += 1

            for suit in LLMAgent._SUITS:
                if suit_dict[suit] == 7:
                    hcp += 3
                elif suit_dict[suit] == 6:
                    hcp += 2
                elif suit_dict[suit] == 5 and (suit == "S" or suit == "H"):
                    hcp += 1

        return hcp, suit_dict

    def make_bid(self, state):
        """
        Make a bidding decision based on the PGX bridge state.

        Args:
            state: PGX bridge state object containing observation and game info

        Returns:
            int: Action index (0-37) corresponding to the chosen bid
        """

        # Extract observation vector
        obs = state.observation  # Shape: (480,) - from state.observation
        
        # === GAME STATE INFORMATION ===
        current_player = state.current_player
        dealer = state._dealer
        shuffled_players = state._shuffled_players
        vul_ns = state._vul_NS
        vul_ew = state._vul_EW
        last_bid = state._last_bid
        last_bidder = state._last_bidder
        call_x = state._call_x
        call_xx = state._call_xx
        legal_actions = state.legal_action_mask
        
        # === BIDDING HISTORY BY CONTRACT ===
        # Each contract has 12 elements (3 per player: bid, double, redouble)
        # obs[8:20]   - Against 1C
        # obs[20:32]  - Against 1D  
        # obs[32:44]  - Against 1H
        # obs[44:56]  - Against 1S
        # obs[56:68]  - Against 1NT
        # ... and so on up to obs[416:428] for 7NT
        
        # === HAND CARDS ===
        hand_cards_vector = obs[428:480].astype(jnp.int32)
        hand_cards = []
        for index in range(len(hand_cards_vector)):
            if hand_cards_vector[index] == 1:
                hand_cards.append(index)
        
        # === PARTNER INFORMATION ===
        partner_index = (current_player + 2) % 4
        partner_physical_position = shuffled_players[partner_index]
        opponent1_index = (current_player + 1) % 4
        opponent2_index = (current_player + 3) % 4

        if last_bid == -1:
            return LLMAgent.opening_bid(hand_cards)

        bidding_history = state._bidding_history
        bidding_history = [x for x in bidding_history if x != -1]

        # Check bidding situation based on history length and last bidder
        if len(bidding_history) == 0:
            # Opening bid - no bids have been made yet
            action = LLMAgent.opening_bid(hand_cards)
        elif len(bidding_history) == 1 and last_bidder == partner_index:
            # Opponent's opening bid
            action = LLMAgent.opponent_opening_bid(bidding_history[0], hand_cards)
        elif len(bidding_history) == 2:
            # Partner's response to opening
            action = LLMAgent.partner_opening_bid(bidding_history[0], bidding_history[1], hand_cards)
        else:
            action = LLMAgent.default_bid(bidding_history, hand_cards)

        # === ACTION SELECTION ===
        return LLMAgent.final_validity_check(action, bidding_history)

    def opening_bid(hand_cards: list[str]):
        hcp, suit_dict = LLMAgent.calc_hcp(hand_cards)

# ==== HELPERS ====


def check_bid_validity(bid_options, bidding_history):
    new_bids = []
    last_bid = bidding_history[-1]
    for bid in bid_options:
        if bid in ["Pass", "Double"]:
            new_bids.append(bid)
        elif bid == "Redouble":
            if len(bidding_history) > 2:
                first_opp_bid = bidding_history[-3]
                partner_bid = bidding_history[-2]
                if first_opp_bid == 1 and partner_bid == 0 and last_bid == 0:
                    new_bids.append(bid)
            elif last_bid == 1:
                new_bids.append(bid)
        else:
            index = 0
            for key, val in LLMAgent._ACTION_IDENTIFIER.items():
                if val == bid:
                    index = key
                    break

            if index > last_bid:
                new_bids.append(bid)

    return new_bids

def get_partner_bid(bidding_history):

    my_index = len(bidding_history) % 4
    partner_index = (my_index + 2) % 4

    for i in range(len(bidding_history) - 1, -1, -1):
        if i % 4 == partner_index:
            bid_value = bidding_history[i]
            if LLMAgent._ACTION_IDENTIFIER[bid_value] not in ["Pass", "Double", "Redouble"]:
                return bid_value

    return None

def final_validity_check(bid, bidding_history):
    highest_bid = max(bidding_history)

    bid_index = 0
    for key, val in LLMAgent._ACTION_IDENTIFIER.items():
        if val == bid:
            bid_index = key
            break

    if bid_index <= 2 or bid_index > highest_bid:
        return bid_index
    else:
        return 0    # index to Pass (default behavior when action isn't legal)