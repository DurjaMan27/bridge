import jax
import jax.numpy as jnp
import numpy as np
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.state import AgentMockState

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    _BID_TO_INDEX = {v: k for k, v in _ACTION_IDENTIFIER.items()}

    # Card index mapping (same as baseline.py)
    _CARD_INDEX = {
        i: f"{['C', 'D', 'H', 'S'][i // 13]}{['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'][i % 13]}" 
        for i in range(52)
    }

    def __init__(self, model_name="gpt-4o"):
        """
        Initializes the OpenAI client and model settings.
        Expects OPENAI_API_KEY to be set in environment variables or .env file.
        """
        api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        if api_key == "YOUR_OPENAI_API_KEY":
            logger.warning("OPENAI_API_KEY not found or is default. LLM calls will fail.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        # All agent calls are assumed to be non-parametric (like baseline)
        self.params = None

    def _format_hand(self, observation: np.ndarray) -> str:
        """Converts the one-hot observation array into a readable hand string."""
        hand_indices = np.where(observation[428:480] == 1)[0]

        # Group cards by suit
        suits = {'C': [], 'D': [], 'H': [], 'S': []}
        for idx in hand_indices:
            card_str = self._CARD_INDEX[idx]
            suit = card_str[0]
            rank = card_str[1:]
            suits[suit].append(rank)

        # Sort ranks and format the string (Spades, Hearts, Diamonds, Clubs)
        formatted_hand = []
        for suit in ['S', 'H', 'D', 'C']:
            if suits[suit]:
                formatted_hand.append(f"{suit}:{''.join(suits[suit])}")
            else:
                formatted_hand.append(f"{suit}:-")

        return " ".join(formatted_hand)

    def _format_bidding_history(self, bidding_history: np.ndarray) -> str:
        """Converts the integer bidding history into a string list of calls."""
        calls = [self._ACTION_IDENTIFIER[int(b)] for b in bidding_history if b != -1]
        return ", ".join(calls)

    # Add this method to LLMAgent class
    def _get_hand_stats(self, observation: np.ndarray) -> tuple[int, str]:
        """Calculates HCP and distribution shape (e.g., '5-3-3-2') from observation."""
        hand_indices = np.where(observation[428:480] == 1)[0]
        
        hcp = 0
        suits = {'S': 0, 'H': 0, 'D': 0, 'C': 0}
        
        for idx in hand_indices:
            # Rank mapping: 0=A, 1=K, 2=Q, 3=J, 4=T, ... (Based on your _CARD_INDEX)
            # CAREFUL: Verify your _CARD_INDEX logic. 
            # In your snippet: 0-12 are Clubs, 13-25 Diamonds, etc.
            # And within 13 cards: 0=A, 1=K... or 0=2, 12=A?
            # Your snippet says: ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'][i % 13]
            # So i % 13 == 0 is Ace (4 pts), 1 is King (3 pts)...
            
            rank_idx = idx % 13
            if rank_idx == 0: hcp += 4
            elif rank_idx == 1: hcp += 3
            elif rank_idx == 2: hcp += 2
            elif rank_idx == 3: hcp += 1
            
            suit_idx = idx // 13
            suit_char = ['C', 'D', 'H', 'S'][suit_idx]
            suits[suit_char] += 1

        # Format shape descending e.g., "5-3-3-2"
        shape_counts = sorted(suits.values(), reverse=True)
        shape_str = "-".join(map(str, shape_counts))
        
        # Detailed shape for prompt (e.g., "5S-3H-3D-2C")
        # useful for evaluating specific suit lengths
        detailed_shape = f"{suits['S']}S-{suits['H']}H-{suits['D']}D-{suits['C']}C"
        
        return hcp, detailed_shape, shape_str

    def _generate_prompt(self, state) -> str:
        """Creates the full prompt for the LLM based on the game state."""
        
        # 1. Player's Hand
        hand_str = self._format_hand(state.observation)
        
        # 2. Bidding History
        history_str = self._format_bidding_history(state._bidding_history)
        hcp, detailed_shape, shape_str = self._get_hand_stats(state.observation)
        
        # 3. Current Vul (for scoring context)
        vul_str = "Both" if state._vul_NS and state._vul_EW else \
                "NS Only" if state._vul_NS else \
                "EW Only" if state._vul_EW else \
                "None"
        
        # 4. Legal Actions
        legal_bids = [
            self._ACTION_IDENTIFIER[i] 
            for i, is_legal in enumerate(state.legal_action_mask) 
            if is_legal
        ]
        legal_bids_str = ", ".join(legal_bids)

        # SYSTEM INSTRUCTION: Define the LLM's role and output format
        system_prompt = (
            "You are a World-Class Bridge Champion playing in a high-stakes IMP tournament. "
            "You and your partner use a MODERN, AGGRESSIVE NATURAL system. Your goal is to maximize EXPECTED IMPs. "

            "STRATEGIC MANDATES:"
            "1. OPENING: Use the 'Rule of 20' (Open if HCP + length of your two longest suits >= 20). Do not wait for 12 HCP if you have shape."
            "2. VULNERABILITY: At IMPs, a vulnerable game is worth the risk. If you have a 40% chance of making, BID IT."
            "3. COMPETITIVE EDGE: Do not let the opponents steal the auction for a part-score. If you have a fit and distribution, compete to the 3-level. "
            "4. OVERCALLS: Be aggressive. A 1-level overcall shows 7-15 points and a 5-card suit. It is a lead-director and a defensive tool. "
            "5. PARTNERSHIP TRUST: Assume your partner (the other LLM) is following these exact aggressive mandates. If they bid, they have 'working' cards. "

            "THOUGHT PROCESS (Chain-of-Thought):"
            "- Evaluate 'Working' Points: Are your honors in your long suits? "
            "- Mental Simulation: If partner has a minimum for their bid, can I visualize 10 tricks (for game) or 7 tricks (for part-score)?"
            "- IMP Math: Is the risk of a penalty smaller than the potential game bonus?"

            "OUTPUT FORMAT (JSON ONLY):"
            "{"
            "   'hand_evaluation': 'Analysis of HCP vs Shape (Rule of 20 check).'"
            "   'auction_interpretation': 'What has partner and the opponents told me about the hidden cards?'"
            "   'mental_simulation': 'Visualizing the play: 'If I bid X, we need Y to happen to make.''"
            "   'risk_reward_ratio': 'Why this bid is mathematically superior for IMPs.'"
            "   'bid': 'EXACT_BID_STRING'"
            "}"
        )

        # USER QUERY: Provide the game context
        user_query = (
            f"**Context**:\n"
            f"  - **Vulnerability**: {vul_str}\n"
            f"  - **Bidding History**: {history_str}\n"
            f"  - **Your Hand**: {hand_str}\n"
            f"  - **Hand Stats**: {hcp} HCP, Distribution: {detailed_shape} ({shape_str})\n" # <--- INSERTED
            f"  - **Legal Bid Options**: {legal_bids_str}\n\n"
            f"What is your next bid? (Respond ONLY with the exact bid string)"
        )
        return system_prompt, user_query

    def make_bid(self, state) -> tuple[int, np.ndarray]:
        """
        Generates a bid by calling the OpenAI API.
        
        The 'state' here is a MockState object created in callback_llm.py or agent_server.py,
        containing NumPy arrays.
        
        Returns: (action_index, pi_probs_array)
        """
        import json  # Ensure json is available
        
        system_prompt, user_query = self._generate_prompt(state)
        # print(system_prompt) # Optional: Un-comment if you still want to see the full prompt
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0,  # Use deterministic output for testing
                response_format={"type": "json_object"} # Enforce JSON output
            )
            
            # 1. Parse the JSON response
            raw_content = completion.choices[0].message.content
            try:
                response_data = json.loads(raw_content)
                reasoning = response_data.get("imp_rationale", "No reasoning provided.")
                bid = response_data.get("bid", "Pass") # Default to Pass if key missing
                
                # 2. Print the reasoning and bid for debugging
                print(f"\n[LLM Reasoning]: {reasoning}")
                print(f"[LLM Bid]: {bid}\n")
                
            except json.JSONDecodeError:
                logger.error(f"LLM returned invalid JSON: {raw_content}. Falling back to 'Pass'.")
                bid = "Pass"

            # 3. Clean up whitespace/quotes (Standard Logic)
            bid = str(bid).replace('"', '').replace("'", '').strip()
            
            # 4. Check if the bid is valid and legal (Standard Logic)
            if bid not in self._BID_TO_INDEX:
                logger.error(f"LLM returned unknown bid: '{bid}'. Falling back to 'Pass'.")
                action_idx = self._BID_TO_INDEX["Pass"]
            else:
                action_idx = self._BID_TO_INDEX[bid]

                # Ensure the chosen action is actually legal
                if not state.legal_action_mask[action_idx]:
                    logger.error(f"LLM chose illegal bid '{bid}'. Legal options: {self._format_bidding_history(state.legal_action_mask)}. Falling back to 'Pass'.")
                    action_idx = self._BID_TO_INDEX["Pass"]

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}. Returning 'Pass' as fallback.")
            action_idx = self._BID_TO_INDEX["Pass"]
        
        # For LLM, we return a one-hot distribution (100% confidence in the chosen bid)
        pi_probs = np.zeros(38, dtype=np.float32)
        pi_probs[action_idx] = 1.0
        
        return action_idx, pi_probs

# Helper function for server/callback to call the bid logic
def llm_bid_from_arrays(
    observation, current_player, legal_action_mask, terminated, rewards,
    last_bid, last_bidder, call_x, call_xx, dealer, shuffled_players,
    vul_NS, vul_EW, bidding_history
) -> tuple[int, np.ndarray]:
    """
    Converts raw numpy arrays back to a MockState and calls the LLM Agent.
    """
    
    # Use the imported, correct class: AgentMockState
    state = AgentMockState(
        observation, current_player, legal_action_mask, terminated, rewards,
        last_bid, last_bidder, call_x, call_xx, dealer, shuffled_players,
        vul_NS, vul_EW, bidding_history
    )
    
    agent = LLMAgent()
    return agent.make_bid(state)