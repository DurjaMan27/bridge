import jax
import jax.numpy as jnp
import numpy as np
import logging
from dotenv import load_dotenv

load_dotenv()

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
    if hand_cards.dtype != jnp.int32:
      hand_cards = hand_cards.astype(jnp.int32)

    # Reshape to (4, 13) for [C, D, H, S] x [A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2]
    hand_reshaped = hand_cards.reshape(4, 13)

    suit_counts = jnp.sum(hand_reshaped, axis=1)

    hcp_values = jnp.array([4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    hcp_per_suit = jnp.sum(hand_reshaped * hcp_values, axis=1)
    base_hcp = jnp.sum(hcp_per_suit)

    # Distributional points
    # 7 cards: +3, 6 cards: +2, 5 cards in S/H: +1
    dist_points = jnp.sum(
      jnp.where(suit_counts == 7, 3,
      jnp.where(suit_counts == 6, 2,
      jnp.where((suit_counts == 5) & (jnp.arange(4) >= 2), 1, 0))))

    total_hcp = base_hcp + dist_points

    return total_hcp, suit_counts


  @staticmethod
  def make_bid(state):
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
      return BaselineAgent.opening_bid(hand_cards)

    bidding_history = state._bidding_history

    # Check bidding situation based on history length and last bidder
    if len(bidding_history) == 0:
      # Opening bid - no bids have been made yet
      action = BaselineAgent.opening_bid(hand_cards)
    elif len(bidding_history) == 1 and last_bidder == partner_index:
      # Opponent's opening bid
      action = BaselineAgent.opponent_opening_bid(bidding_history[0], hand_cards)
    elif len(bidding_history) == 2:
      # Partner's response to opening
      action = BaselineAgent.partner_opening_bid(bidding_history[0], bidding_history[1], hand_cards)
    else:
      action = BaselineAgent.default_bid(bidding_history, hand_cards)

    # === ACTION SELECTION ===
    return BaselineAgent.final_validity_check(action, bidding_history)

  def opening_bid(hand_cards: list[str]):
    hcp, suit_dict = BaselineAgent.calc_hcp(hand_cards)

    opening_options = BaselineAgent.majors_and_minors(suit_dict)

    if hcp >= 13 and hcp < 21:
      if hcp >= 16 and hcp <= 18 and opening_options["1NT"]:
        return "1NT"
      else:
        if opening_options["1H/1S"]:
          return opening_options["1H/1S"]
        elif opening_options["1C/1D"]:
          return opening_options["1C/1D"]
    elif hcp >= 21 and hcp <= 23 and opening_options["2NT"]:
      return "2NT"
    elif hcp >= 22 and opening_options["2_suit"]:
      return opening_options["2_suit"]
    elif hcp >= 5 and hcp < 9 and opening_options["3_suit"]:
      return opening_options["3_suit"]
    elif hcp >= 6 and hcp <= 10 and opening_options["4_suit"]:
      return opening_options["4_suit"]
    
    return "Pass"
  
  def opponent_opening_bid(opening_bid, hand_cards):
    hcp, suit_dict = BaselineAgent.calc_hcp(hand_cards)
    spades = suit_dict["S"]
    hearts = suit_dict["H"]
    diamonds = suit_dict["D"]
    clubs = suit_dict["C"]

    if opening_bid < 3:
      opener_suit = BaselineAgent._ACTION_IDENTIFIER[opening_bid]
    else:
      opener_suit = BaselineAgent._SUITS[(opening_bid - 3) % 4]

    candidate_bids = []

    # 1. Weak Hand
    if hcp <= 7:
      candidate_bids.append("Pass")

    # 2. Takeout Double
    unbid_suits = [s for s in ["C", "D", "H", "S"] if s != opener_suit]
    if hcp >= 12 and all(suit_dict[s] >= 3 for s in unbid_suits):
      candidate_bids.append("Double")

    # 3. 1NT Overcall
    if 15 <= hcp <= 18:
      balanced = all(count <= 5 for count in suit_dict.values())
      if balanced:
        candidate_bids.append("1NT")

    # 4. Simple Suit Overcall
    for suit, length in suit_dict.items():
      if suit != opener_suit and 8 <= hcp <= 16 and length >= 5:
        candidate_bids.append(f"1{suit}")

    # 5. Jump Overcall
    for suit, length in suit_dict.items():
      if suit != opener_suit and 6 <= hcp <= 10 and length >= 6:
        candidate_bids.append(f"2{suit}")

    # 6. Preemptive Higher-Level Overcall
    for suit, length in suit_dict.items():
      if suit != opener_suit and hcp < 10 and length >= 7:
        candidate_bids.append(f"3{suit}")

    # 7. Cue Bid (very strong hand, 18+)
    if hcp >= 18:
      candidate_bids.append(f"2{opener_suit}")

    legal_actions = BaselineAgent.check_bid_validity(candidate_bids, [opening_bid])

    # Default
    return "Pass" if len(legal_actions) == 0 else legal_actions[0]

  def partner_opening_bid(opening_bid, opp_bid, hand_cards):
    hcp, suit_dict = BaselineAgent.calc_hcp(hand_cards)
    spades = suit_dict["S"]
    hearts = suit_dict["H"]
    diamonds = suit_dict["D"]
    clubs = suit_dict["C"]

    candidate_bids = []

    # Only supporting 1NT opening logic here
    if opening_bid == 7:

      # 1. Weak hands: sign-offs
      if hcp <= 7:
        if hearts >= 5:
          candidate_bids.append("2H")
        elif spades >= 5:
          candidate_bids.append("2S")
        elif diamonds >= 5:
          candidate_bids.append("2D")
        else:
          candidate_bids.append("Pass")

      # 2. Stayman (looking for 4-4 major fit)
      if 8 <= hcp <= 9 and (hearts >= 4 or spades >= 4):
        candidate_bids.append("2C")

      # 3. Invitational to 3NT
      if 8 <= hcp <= 9:
        candidate_bids.append("2NT")

      # 4. Game-level bids
      if 10 <= hcp <= 15:
        # Balanced hand → 3NT
        if max(spades, hearts) < 5:
          candidate_bids.append("3NT")
        # With 5-card major, force with 3-of-a-suit
        if spades >= 5:
          candidate_bids.append("3S")
        if hearts >= 5:
          candidate_bids.append("3H")

      # 5. Slam tries
      if 16 <= hcp <= 18:
        candidate_bids.append("4NT")  # invitational to 6NT

      if hcp >= 19:
        candidate_bids.append("4C")   # Gerber, asking for aces

      legal_actions = BaselineAgent.check_bid_validity(candidate_bids, [opening_bid, opp_bid])
      
      return "Pass" if len(legal_actions) == 0 else legal_actions[0]

    elif opening_bid in [3, 4, 5, 6]:

      opener_suit = BaselineAgent._ACTION_IDENTIFIER[opening_bid][1:]
      opener_length = suit_dict[opener_suit]

      # Assume opener has at least 5 in their suit
      total_trumps = opener_length + 5

      candidate_bids = []

      # 1. Weak hand (0–5 HCP)
      if hcp <= 5:
        candidate_bids.append("Pass")

      # 2. Supporting partner’s suit
      if suit_dict[opener_suit] >= 3:  
        # Simple raise: 6–10 pts + 8+ total trumps
        if 6 <= hcp <= 10 and total_trumps >= 8:
          candidate_bids.append(f"2{opener_suit}")

        # Jump raise: 13+ pts + support
        if hcp >= 13:
          candidate_bids.append(f"3{opener_suit}")

        # Double-jump sign-off: weak but many trumps
        if hcp <= 9 and total_trumps >= 10:
          candidate_bids.append(f"4{opener_suit}")

      # 3. One-over-one response (new suit at 1-level, 6–18 pts, 4+ cards)
      # Legal if the suit bid is higher ranking than opener's suit at 1-level
      for suit, length in suit_dict.items():
        if length >= 4 and 6 <= hcp <= 18:
          if suit in ["D", "H", "S"]:  # must be a higher-ranking suit
            if opening_bid == "1C" and suit in ["D", "H", "S"]:
              candidate_bids.append(f"1{suit}")
            if opening_bid == "1D" and suit in ["H", "S"]:
              candidate_bids.append(f"1{suit}")
            if opening_bid == "1H" and suit == "S":
              candidate_bids.append("1S")

      # 4. Two-over-one (10–18 pts, 4+ in new suit, non-jump)
      for suit, length in suit_dict.items():
        if length >= 4 and 10 <= hcp <= 18:
          # must be a non-jump at 2-level, not opener's suit
          if suit != opener_suit:
            candidate_bids.append(f"2{suit}")

      # 5. Jump shift (19+ pts, new suit)
      for suit, length in suit_dict.items():
        if length >= 4 and hcp >= 19 and suit != opener_suit:
          candidate_bids.append(f"2{suit}")  # jump-shift (forcing)

      # 6. NT responses
      if 6 <= hcp <= 9:
        candidate_bids.append("1NT")
      if 13 <= hcp <= 15:
        candidate_bids.append("2NT")
      if 16 <= hcp <= 18:
        candidate_bids.append("3NT")

      legal_actions = BaselineAgent.check_bid_validity(candidate_bids, [opening_bid, opp_bid])
      
      return "Pass" if len(legal_actions) == 0 else legal_actions[0]

  def default_bid(bidding_history, hand_cards):
    hcp, suit_dict = BaselineAgent.calc_hcp(hand_cards)

    candidate_bids = []
    last_bid = bidding_history[-1]
    last_suit = "" if last_bid < 3 else BaselineAgent._ACTION_IDENTIFIER[last_bid][1:]

    # 1. Pass if weak
    if hcp < 6:
      candidate_bids.append("Pass")

    # 2. Support partner's suit
    partner_bid = BaselineAgent.get_partner_bid(bidding_history)
    
    if partner_bid:
      level, suit = BaselineAgent._ACTION_IDENTIFIER[last_bid][0], BaselineAgent._ACTION_IDENTIFIER[last_bid][1:]
      support = suit_dict.get(suit, 0)

      if 6 <= hcp <= 10 and support >= 3:
        proposed = f"{level+1}{suit}"
        candidate_bids.append(proposed)

      if hcp >= 13 and support >= 3:
        proposed = f"{level+2}{suit}"
        candidate_bids.append(proposed)

      if hcp <= 10 and support + level >= 8:
        proposed = f"{level + 3}{suit}"
        candidate_bids.append(proposed)

    # 3. New suit bids (one-over-one or jump shift)
    for s, count in suit_dict.items():
      if partner_bid and s == partner_bid[1]:
        continue

      if count >= 4 and 6 <= hcp <= 18:
        proposed = f"1{s}" if last_bid is None else f"{int(last_bid[0])+0}{s}"
        candidate_bids.append(proposed)

      if count >= 4 and hcp >= 19:
        proposed = f"2{s}"
        candidate_bids.append(proposed)

    # 4. NT responses
    balanced = all(c <= 5 for c in suit_dict.values())
    if balanced:
      if 6 <= hcp <= 9:
        candidate_bids.append("1NT")
      if 13 <= hcp <= 15:
        candidate_bids.append("2NT")
      if 16 <= hcp <= 18:
        candidate_bids.append("3NT")

    # 5. Takeout double
      # if last bid is by opponent and you have support for unbid suits
    if last_bid and last_bid not in [0, 1, 2]:
      unbid_suits = [s for s in ["C", "D", "H", "S"] if s != last_suit and suit_dict.get(s, 0) >= 3]
      if hcp >= 12 and unbid_suits:
        candidate_bids.append("Double")
        
    # 6. Redouble (if partner or you were doubled)
    if "Double" in bidding_history[-1:]:
      candidate_bids.append("Redouble")

    candidate_bids.append("Pass")

    legal_actions = BaselineAgent.check_bid_validity(candidate_bids, bidding_history)
      
    return "Pass" if len(legal_actions) == 0 else legal_actions[0]

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

    for i in range(len(bidding_history)-1, -1, -1):
      if i % 4 == partner_index:
        if BaselineAgent._ACTION_IDENTIFIER[bidding_history[i]] not in ["Pass", "Double", "Redouble"]:
          return bidding_history[i]

    return None

  def final_validity_check(bid, bidding_history):
    highest_bid = max(bidding_history)

    bid_index = 0
    for key, val in BaselineAgent._ACTION_IDENTIFIER.items():
      if val == bid:
        bid_index = key
        break

    if bid <= 2 or bid > highest_bid:
      return bid_index
    else:
      return 0    # index to Pass (default behavior when action isn't legal)