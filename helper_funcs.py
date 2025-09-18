import math
import random
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

def queryLLM(player, partner, cards, bid_history):
  messages = [
      SystemMessage(
          content="You are a world champion Bridge player that plays consistent, always in touch with your partner's decisions."
      )
  ]

  human = HumanMessage(content=
        f"""You will be given your 13-card hand for Bridge. You are player {player}. Players bid in the order N → E → S → W, with N opening the bidding. You are partners with {partner}.
            Here is your hand: {cards}

            Here is the full bidding history so far: {bid_history}

            Use this information to make your next bid. Take into account:
            - Your high card points (HCP)
            - Your hand distribution (balanced vs. unbalanced)
            - Whether you or your partner has already bid
            - The meaning of your partner's bid if any (showing strength or suit preference)
            - Whether opponents have doubled, preempted, or passed
            - Reasonable conventions such as opening 1NT with 15–17 balanced, or doubling with 12+ points and no suit to bid
            - Whether you are in first, second, third, or fourth seat
            - Your strategy: you play consistent, partnership-driven Bridge — not overly aggressive or wildly conservative

            Do not offer any explanation.
            Respond only with a single legal Bridge bid in string format (e.g. '1H', 'Pass', 'Double', '1NT', 'Redouble') in the following format:

            FINAL BID: [bid]
            """
        )
  
  messages.append(human)

  llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

  # Instantiate a chat model and invoke it with the messages
  response = llm.invoke(messages)
  return response

def within_range(under, over, num):
  return under <= num and num <= over

def divide_by_suit(cards):
  suit_dict = {"C": 0, "D": 0, "H": 0, "S": 0}

  for card in cards:
    value, suit = card[:len(card) - 1], card[len(card) - 1:]
    suit_dict[suit] += 1

  return suit_dict

def balanced(suit_dict):
  totals = []
  for suit in ["C", "D", "H", "S"]:
    totals.append(suit_dict[suit])

  if sorted(totals) == [3, 3, 3, 4]:
    return True
  elif sorted(totals) == [2, 3, 4, 4]:
    return True
  elif sorted(totals) == [2, 3, 3, 5] and suit_dict["S"] != 5 and suit_dict["H"] != 5:
    return True
  else:
    return False

def greater_than(dict, baseline):
  for value in dict.values():
    if value >= baseline:
      return True
    
  return False