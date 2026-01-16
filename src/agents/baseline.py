import jax
import jax.numpy as jnp
import numpy as np
import logging
from dotenv import load_dotenv

load_dotenv()

import jax.numpy as jnp

class BaselineAgent():
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

    _SUITS = ["C", "D", "H", "S", "NT"]

    def calc_hcp(hand_cards):
        hcp = 0
        suit_counts = {"C": 0, "D": 0, "H": 0, "S": 0}
        
        for card_idx in hand_cards:
            val = card_idx % 13
            suit = ["C", "D", "H", "S"][card_idx // 13]
            suit_counts[suit] += 1
            if val == 0: hcp += 4   # Ace
            elif val == 12: hcp += 3 # King
            elif val == 11: hcp += 2 # Queen
            elif val == 10: hcp += 1 # Jack

        points = hcp
        for s in ["C", "D", "H", "S"]:
            if suit_counts[s] == 5: points += 1
            elif suit_counts[s] == 6: points += 2
            elif suit_counts[s] >= 7: points += 3
        
        return points, hcp, suit_counts

    def _get_bid_info(self, bid_idx):
        if bid_idx < 3: 
            return None, None
        level = (bid_idx - 3) // 5 + 1
        suit_idx = (bid_idx - 3) % 5
        return level, self._SUITS[suit_idx]

    def _to_idx(self, level, suit_str):
        s_idx = self._SUITS.index(suit_str)
        return 3 + (level - 1) * 5 + s_idx

    def make_bid(self, state):
        obs = state.observation
        hand_cards = jnp.where(obs[428:480] == 1)[0].tolist()
        total_pts, hcp, suits = BaselineAgent.calc_hcp(hand_cards)
        
        history = [int(x) for x in state._bidding_history if x != -1]
        my_pos = len(history) % 4
        
        last_real_bid = -1
        last_bidder = -1
        for i in range(len(history)-1, -1, -1):
            if history[i] >= 3:
                last_real_bid = history[i]
                last_bidder = i % 4
                break

        if last_real_bid == -1:
            bid = self.opening_bid(total_pts, hcp, suits)
        else:
            rel_pos = (my_pos - last_bidder) % 4
            if rel_pos == 2: # Partner bid last
                bid = self.respond_to_partner(last_real_bid, total_pts, hcp, suits)
            else: # Opponent bid last
                bid = self.overcall(last_real_bid, total_pts, hcp, suits)

        if not state.legal_action_mask[bid]:
            # Fallback to Pass if illegal
            return 0 if state.legal_action_mask[0] else jnp.argmax(state.legal_action_mask)
        
        return bid

    def opening_bid(self, pts, hcp, suits):
        is_balanced = sorted(suits.values()) in [[3,3,3,4], [2,3,3,5], [2,3,4,4]]
        no_5_major = suits["H"] < 5 and suits["S"] < 5

        if 16 <= hcp <= 18 and is_balanced and no_5_major: return 7 # 1NT
        if 21 <= hcp <= 23 and is_balanced: return 12 # 2NT
        if pts >= 22:
            for s in ["S", "H", "D", "C"]:
                if suits[s] >= 5:
                    return self._to_idx(2, s)
        if 13 <= pts <= 21:
            if suits["S"] >= 5 or suits["H"] >= 5:
                return self._to_idx(1, "S" if suits["S"] >= suits["H"] else "H")
            if suits["D"] >= 4 or suits["C"] >= 4:
                return self._to_idx(1, "D" if suits["D"] >= suits["C"] else "C")
            return self._to_idx(1, "C") # 3-3 minors open 1C
        if 5 <= hcp <= 9:
            for s in ["S", "H", "D", "C"]:
                if suits[s] == 7:
                    return self._to_idx(3, s)
        return 0

    # def overcall(self, last_bid, pts, hcp, suits):
    #     lvl, suit = self._get_bid_info(last_bid)
    #     if 16 <= hcp <= 18 and all(c <= 5 for c in suits.values()):
    #         return 7 # 1NT
    #     if pts >= 13: # Takeout Double logic simplified
    #         unbid = [s for s in ["S", "H", "D", "C"] if s != suit and suits[s] >= 3]
    #         if len(unbid) >= 3:
    #             return 1
    #     for s in ["S", "H", "D", "C"]:
    #         if suits[s] >= 5 and 10 <= pts <= 17:
    #             idx = self._to_idx(lvl if s > suit or lvl > 1 else lvl + 1, s)
    #             if idx > last_bid:
    #                 return idx
    #     return 0

    def overcall(self, last_bid, pts, hcp, suits):
        lvl, suit = self._get_bid_info(last_bid)
        
        # Don't overcall above 4-level without strong hand
        if lvl >= 4 and pts < 18:
            return 0
        
        # Don't overcall at 7-level unless you have slam values
        if lvl >= 6 and pts < 25:
            return 0
        
        if 16 <= hcp <= 18 and all(c <= 5 for c in suits.values()):
            if lvl <= 3:  # Only 1NT overcall at low levels
                return 7
        
        for s in ["S", "H", "D", "C"]:
            if suits[s] >= 5 and 10 <= pts <= 17:
                # Calculate minimum level needed
                if suit and s == suit:
                    continue  # Don't overcall in opponent's suit
                
                min_lvl = lvl + 1 if suit and self._SUITS.index(s) <= self._SUITS.index(suit) else lvl
                
                # Cap overcalls at 3-level unless very strong
                if min_lvl > 3 and pts < 17:
                    continue
                    
                idx = self._to_idx(min_lvl, s)
                if idx > last_bid:
                    return idx
        
        return 0

    def respond_to_partner(self, p_bid, pts, hcp, suits):
        lvl, s = self._get_bid_info(p_bid)
        
        if s == "NT":
            if pts <= 7:
                return 0
            if pts <= 9:
                return 12 if lvl == 2 else 11 # Invite
            if 10 <= pts <= 15:
                return self._to_idx(lvl + 1, "NT")
            return 0

        # Support Points adjustment
        support_pts = pts
        if suits[s] >= 3:
            for sn in ["S", "H", "D", "C"]:
                if suits[sn] == 2:
                    support_pts += 1
                elif suits[sn] == 1:
                    support_pts += 2
                elif suits[sn] == 0:
                    support_pts += 3

        if suits[s] >= 3:
            if 6 <= support_pts <= 10:
                target_level = min(lvl + 1, 4)  # Cap at game
                return self._to_idx(target_level, s)
            if 11 <= support_pts <= 12:
                target_level = min(lvl + 2, 4)  # Invite game
                return self._to_idx(target_level, s)
            if 13 <= support_pts <= 15:
                # Game values - bid game
                if s in ["H", "S"]:
                    return self._to_idx(4, s)  # 4H/4S
                else:
                    return self._to_idx(5, s)  # 5C/5D
        
        if 6 <= pts <= 18:
            for sn in ["S", "H", "D", "C"]:
                if suits[sn] >= 4 and sn != s:
                    idx = self._to_idx(1, sn)
                    if idx > p_bid:
                        return idx
        
        if 6 <= pts <= 9:
            return self._to_idx(1, "NT") if p_bid < 7 else 0
        return 0

    # ==== HELPERS ====
    def majors_and_minors(suit_dict):
        """
        suit_dict: dict like {"S": 5, "H": 3, "D": 3, "C": 2}
        Returns: dict mapping bid type -> string or None
        """
        result = {
            "1NT": None,
            "1H/1S": None,
            "1C/1D": None,
            "2NT": None,
            "2_suit": None,
            "3_suit": None,
            "4_suit": None,
        }

        suits = ["S", "H", "D", "C"]
        values = [suit_dict[s] for s in suits]
        sorted_vals = sorted(values, reverse=True)


        balanced_shapes = [
            [4,3,3,3],
            [4,4,3,2],
            [5,3,3,2]
        ]
        is_balanced = sorted_vals in balanced_shapes

        # --- 1NT ---
        if is_balanced and suit_dict["S"] < 5 and suit_dict["H"] < 5:
            result["1NT"] = "1NT"

        # --- 2NT ---
        if is_balanced and suit_dict["S"] < 5 and suit_dict["H"] < 5:
            result["2NT"] = "2NT"

        # --- 1H or 1S ---
        if suit_dict["S"] >= 5 or suit_dict["H"] >= 5:
            if suit_dict["S"] >= 5:
                result["1H/1S"] = "1S"
            else:
                result["1H/1S"] = "1H"

        # --- 1C or 1D ---
        if suit_dict["D"] > suit_dict["C"]:
            result["1C/1D"] = "1D"
        elif suit_dict["C"] > suit_dict["D"]:
            result["1C/1D"] = "1C"
        else:  # equal minors
            if suit_dict["C"] == 3 and suit_dict["D"] == 3:
                result["1C/1D"] = "1C"
            elif suit_dict["C"] >= 4:  # 4-4 or longer
                result["1C/1D"] = "1D"

        # --- 2 of a suit ---
        for s in suits:
            if suit_dict[s] >= 5:
                result["2_suit"] = f"2{s}"
                break

        # --- 3 of a suit ---
        for s in suits:
            if suit_dict[s] == 7:
                result["3_suit"] = f"3{s}"
                break

        # --- 4 of a suit ---
        for s in suits:
            if suit_dict[s] >= 8:
                result["4_suit"] = f"4{s}"
                break

        return result


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
                for key, val in BaselineAgent._ACTION_IDENTIFIER.items():
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
            if BaselineAgent._ACTION_IDENTIFIER[bid_value] not in ["Pass", "Double", "Redouble"]:
                return bid_value

        return None

    def final_validity_check(bid, bidding_history):
        highest_bid = max(bidding_history)

        bid_index = 0
        for key, val in BaselineAgent._ACTION_IDENTIFIER.items():
            if val == bid:
                bid_index = key
                break

        if bid_index <= 2 or bid_index > highest_bid:
            return bid_index
        else:
            return 0    # index to Pass (default behavior when action isn't legal)