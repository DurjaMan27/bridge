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
        hand_indices = np.where(observation[0:52] == 1)[0]
        
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

    def _generate_prompt(self, state) -> str:
        """Creates the full prompt for the LLM based on the game state."""
        
        # 1. Player's Hand
        hand_str = self._format_hand(state.observation)
        
        # 2. Bidding History
        history_str = self._format_bidding_history(state._bidding_history)
        
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
            "You are an expert contract bridge player acting as a Bidding Agent. "
            "Your primary objective is to select the optimal bid that maximizes the expected IMP score. "
            "Rather than adhering to a single rigid system, apply expert bridge logic to deduce the "
            "specific bidding systems and point ranges being employed by both your partner and your "
            "opponents based on the auction history. "
            "Analyze the current vulnerability, your partner's implied holdings, the opponents' "
            "tactical actions, and your 13-card hand (including distribution and high-card strength). "
            "Adjust your strategy dynamically to exploit opponent tendencies and support your partner. "
            "You MUST respond ONLY with the single, exact, case-sensitive bid string "
            "(e.g., '1NT', '3S', 'Pass', 'Double', 'Redouble'). "
            "Do NOT output any additional text, reasoning, markdown formatting, or punctuation. "
            "Your output must be the pure, unadulterated bid string."
        )

        # USER QUERY: Provide the game context
        user_query = (
            f"**Context**:\n"
            f"  - **Vulnerability**: {vul_str}\n"
            f"  - **Bidding History (Past Calls)**: {history_str}\n"
            f"  - **Your Current Hand**: {hand_str}\n"
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
        
        system_prompt, user_query = self._generate_prompt(state)
        print(system_prompt)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0  # Use deterministic output for testing
            )
            
            # Extract the bid, clean up whitespace/quotes
            raw_bid = completion.choices[0].message.content.strip()
            bid = raw_bid.replace('"', '').replace("'", '').strip()
            
            # Check if the bid is valid and legal
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