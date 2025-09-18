import math
import random
import os, re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from helper_funcs import divide_by_suit, within_range, balanced, greater_than, queryLLM

load_dotenv()

class AIAgent():
  def __init__(self, cards):
    self.HIGH_CARDS = ["A", "K", "Q", "J"]
    self.SUITS = ["C", "D", "H", "S"]
    self.cards = cards

  def make_bid(self, player: str, partner: str, bid_history: list[str]):
    response = queryLLM(player, partner, self.cards, bid_history)

    pattern = r"FINAL BID: (.*)"
    match = re.search(pattern, response.content)

    if match:
        return match.group(1).strip()
    else:
        return "Pass"

class BaselineAgent():
  def __init__(self, cards):
    self.HIGH_CARDS = ["A", "K", "Q", "J"]
    self.SUITS = ["C", "D", "H", "S"]
    self.cards = cards
    self.hcp = self.calc_hcp(self.cards)

  def calc_hcp(self, cards):
    hcp = 0
    suit_dict = {"C": 0, "D": 0, "H": 0, "S": 0}

    for card in cards:
      value, suit = card[:len(card)-1], card[len(card)-1:]
      suit_dict[suit] += 1
      for index, high in enumerate(self.HIGH_CARDS):
        if value == high:
          hcp += (4 - index)

    for suit in self.SUITS:
      if suit_dict[suit] == 7:
        hcp += 3
      elif suit_dict[suit] == 6:
        hcp += 2
      elif suit_dict[suit] == 5 and (suit == "S" or suit == "H"):
        hcp += 1

    # -----
    # figure out how to add trump fit
    # -----

    return hcp


  def get_opening_bid(self, cards):
    suit_dict = divide_by_suit(cards)
    if within_range(13, 21, self.hcp):
      if within_range(16, 18, self.hcp) and balanced(suit_dict):
        return "1NT"
      elif suit_dict["H"] >= 5 or suit_dict["S"] >= 5:
        if suit_dict["S"]:
          return "1S"
        else:
          return "1H"
      elif suit_dict["C"] >=3 and suit_dict["D"] >= 3:
        if suit_dict["C"] > suit_dict["D"]:
          return "1C"
        elif suit_dict["C"] < suit_dict["D"]:
          return "1D"
        elif suit_dict["C"] == 3:
          return "1C"
        else:
          return "1D"
    elif within_range(21, 23, self.hcp) and balanced(suit_dict):
      return "2NT"
    elif self.hcp >= 22 and (suit_dict["H"] >= 5 or suit_dict["S"] >= 5 or suit_dict["C"] >= 5 or suit_dict["D"] >= 5):
      return "2" + max(suit_dict, key=suit_dict.get)
    elif within_range(6, 10, self.hcp) and greater_than(suit_dict, 8):
      return "4" + max(suit_dict, key=suit_dict.get)
    elif within_range(5, 9, self.hcp) and greater_than(suit_dict, 7):
      return "3" + max(suit_dict, key=suit_dict.get)
    else:
      return "pass"

  def opponent_bid(self, last_bid, last_bidder_position, my_position):
    if last_bid is None:
        # No prior bid to respond to — this is an opening bid
        if 15 <= self.hcp <= 17:
            return '1NT'
        elif self.hcp >= 13:
            return '1C'  # or 1D depending on distribution
        else:
            return 'Pass'

    # Only consider doubling if the last bid was by an opponent AND was a real bid (not 'Pass')
    is_opponent = (last_bidder_position % 2) != (my_position % 2)
    if is_opponent and last_bid != 'Pass':
        if self.hcp >= 13 and ('NT' in last_bid or last_bid[1] in 'CDHS'):
            return 'Double'

    # Non-double responses
    if 10 <= self.hcp < 13:
        return '1NT' if last_bid != 'Pass' and last_bid < '1NT' else 'Pass'
    else:
        return 'Pass'


  def partner_bid(self, partner_bid, was_doubled=False):
    suit = partner_bid[1:]
    level = int(partner_bid[0])

    if was_doubled and self.hcp >= 10:
      return 'Redouble'
    elif self.hcp < 6:
      return 'Pass'
    elif 6 <= self.hcp <= 9:
      return f'{level + 1}{suit}'
    elif 10 <= self.hcp <= 12:
      return f'{level + 2}{suit}'
    elif self.hcp >= 13:
      return f'4{"" if suit == "NT" else suit}'

    return 'Pass'
  
  def get_partner(self, index):
    return (index + 2) % 4

  def is_doubled_against_you(self, bid_history, current_team):
    for i in range(len(bid_history)-1, -1, -1):
      bid = bid_history[i]
      if bid in ['Double', 'Redouble']:
        return (i % 2) != current_team  # doubled by opponent
      if bid not in ['Pass']:
        break
    return False


class BridgeBidding():

  def __init__(self):
    self.bid_history = []
    self.hands = []         # dealer first, then clockwise
    self.turn = 1           # turn 1 is for dealer, cycles from 1 to 4
    self.agents: list[BaselineAgent] = []

    self.generate_hands()
    self.create_agents()

  def generate_hands(self):
    SUITS = ['C', 'D', 'H', 'S']
    VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    deck = list(range(52))
    random.shuffle(deck)

    cards = [[], [], [], []]
    for i in range(52):
        player = i % 4
        cards[player].append(f'{VALUES[deck[i] % 13]}{SUITS[deck[i] // 13]}')

    self.hands = cards
    print(f"Hands Below:\nNorth: {cards[0]}\nEast: {cards[1]}\nSouth: {cards[2]}\nWest: {cards[3]}")

  def create_agents(self):
    temp = []
    for index in range(4):
      temp.append(BaselineAgent(self.hands[index]))

    self.agents = temp

  def generate_all_bids(self):
    SUITS = ['C', 'D', 'H', 'S', 'NT']
    LEVELS = ['1', '2', '3', '4', '5', '6', '7']
    return [level + suit for level in LEVELS for suit in SUITS]

  def generate_legal_bids(self, bid_history):
    all_bids = self.generate_all_bids()

    # Find the last non-pass bid
    last_bid = None
    for bid in reversed(bid_history):
      if bid != 'Pass':
        last_bid = bid
        break

    if last_bid is None:
      return ['Pass'] + all_bids  # All bids are legal if no one has bid yet
    else:
      last_index = all_bids.index(last_bid)
      return ['Pass'] + all_bids[last_index + 1:]  # Only higher bids allowed


  def simulate_bidding(self):
    bid_history = []
    players = ['N', 'E', 'S', 'W']
    self.turn = 0
    passes_in_a_row = 0

    while True:
      agent = self.agents[self.turn]
      hcp = agent.hcp

      partner_index = agent.get_partner(self.turn)

      # Get the last non-pass bid and its position
      last_bid_info = next(((i, b) for i, b in reversed(list(enumerate(bid_history))) if b not in ['Pass']), (None, None))
      last_bidder_position, last_bid = last_bid_info

      # Find most recent partner bid (excluding passes)
      partner_bid = next(
        (b for i, b in reversed(list(enumerate(bid_history)))
        if i % 2 == partner_index % 2 and b not in ['Pass']),
        None
      )

      was_doubled = agent.is_doubled_against_you(bid_history, self.turn % 2)

    # Decide what kind of bid this player should make
      if not bid_history:
        bid = agent.get_opening_bid(self.hands[self.turn])
        print("OPENING BID!!", bid)
      elif self.turn % 2 == (len(bid_history) - 1) % 2:
        bid = agent.partner_bid(hcp, partner_bid, was_doubled)
      else:
        bid = agent.opponent_bid(last_bid, last_bidder_position, self.turn)

      if bid in self.generate_legal_bids(bid_history):
        bid_history.append(bid)
      else:
        bid_history.append('Pass')
      print(f"{players[self.turn]}: {bid}")

      # Count passes
      if bid == 'Pass':
        passes_in_a_row += 1
      else:
        passes_in_a_row = 0

      if len(bid_history) >= 4 and passes_in_a_row >= 3:
        break

      self.turn = (self.turn + 1) % 4

    return bid_history