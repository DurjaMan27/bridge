import jax
import jax.numpy as jnp
import numpy as np
import logging
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.state import AgentMockState
from src.utils.lps_tracker import lps_tracker

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DiagnosticLogger:
    @staticmethod
    def card_tensor_to_str(hand_array):
        """Converts one-hot or index based card arrays to readable strings.
        Assumes standard 52-card encoding if raw."""
        # Note: This depends on your specific encoding in pgx/bridge.
        # This is a generic placeholder. If you have a specific util in src, use that.
        # For now, we will log the raw array if decoding fails, but let's try a standard mapping.
        ranks = "23456789TJQKA"
        suits = "CDHS"
        try:
            # Assuming boolean array of shape (52,) or (4, 13) where 1 is held
            flat_hand = hand_array.flatten()
            cards = []
            for i, held in enumerate(flat_hand):
                if held:
                    suit = suits[i // 13]
                    rank = ranks[i % 13]
                    cards.append(f"{rank}{suit}")
            return " ".join(cards)
        except Exception:
            return str(hand_array)

    @staticmethod
    def log_turn(state, prompt_messages, llm_response_raw, parsed_bid):
        """Appends a full state dump to a JSONL file."""
        
        # 1. Decode History
        history_str = [str(x) for x in state._bidding_history if x != -1] # Simplify based on your padding
        
        # 2. Parse the JSON response from LLM (if it failed, log raw)
        try:
            # Clean markdown code blocks if present
            clean_content = llm_response_raw.replace("```json", "").replace("```", "").strip()
            reasoning_json = json.loads(clean_content)
        except:
            reasoning_json = {"error": "JSON Parse Fail", "raw_output": llm_response_raw}

        # 3. Construct the Log Entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "turn_index": int(len(history_str)),
            "vulnerability": {
                "NS": bool(state._vul_NS),
                "EW": bool(state._vul_EW)
            },
            "auction_history": history_str,
            "legal_actions": [i for i, x in enumerate(state.legal_action_mask) if x],
            # We assume current_player is an int index (0-3)
            "current_player": int(state.current_player),
            # Attempt to capture the raw hand input. 
            # Note: 'state.observation' usually contains the hand. 
            # We log the shape/raw data for now to ensure we have it.
            "observation_raw_summ": str(np.sum(state.observation)), 
            "llm_full_response": reasoning_json,
            "final_selected_bid": parsed_bid
        }

        # 4. Append to file (Thread safe-ish for small volumes)
        with open("src/logs/llm_diagnostic_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

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
        self.system_prompt = (
            "You are an expert contract bridge player acting as a Bidding Agent. "
            "Your primary objective is to select the optimal bid that maximizes the expected IMP score. "
            "Rather than adhering to a single rigid system, apply expert bridge logic to deduce the "
            "specific bidding systems and point ranges being employed by both your partner and your "
            "opponents based on the auction history. "
            "Analyze the current vulnerability, your partner's implied holdings, the opponents' "
            "tactical actions, and your 13-card hand (including distribution and high-card strength). "
            "Adjust your strategy dynamically to exploit opponent tendencies and support your partner. "
            "You must prioritize reaching a 'Makeable' contract. "
            "Do not jump to a Small Slam (6-level) or Grand Slam (7-level) unless you have deduced that your partnership holds at least 33-37 combined High Card Points. "
            "Avoid bidding at the 7-level if the opponents have shown strength, as a Doubled penalty is catastrophic. "
            "Do not bid at the 5, 6, or 7 level unless your partner has shown significant strength (e.g., via a Jump Shift or a Cue Bid). "
            "A penalty of -2000 points is far worse than missing a slam. "
            "\nOUTPUT FORMAT (JSON ONLY):\n"
            "{{\n"
            "  'context': 'Opening side or Defending side?',\n"
            "  'hand_evaluation': 'HCP, Quick Tricks, Shape, Seat, Vulnerability',\n"
            "  'system_state': 'Auction interpretation with HCP ranges',\n"
            "  'strategy_chosen': 'Which specific strategy above applies?',\n"
            "  'mental_simulation': 'If I bid X, likely outcomes given context',\n"
            "  'bid': 'EXACT_BID_STRING'\n"
            "}}"
        )

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
        return user_query

    def make_bid(self, state) -> tuple[int, np.ndarray]:
        """
        Generates a bid by calling the OpenAI API.
        
        The 'state' here is a MockState object created in callback_llm.py or agent_server.py,
        containing NumPy arrays.
        
        Returns: (action_index, pi_probs_array)
        """
        
        user_query = self._generate_prompt(state)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # --- DIAGNOSTIC LOGGING ---
            parsed_content = json.loads(content)
            bid_str = parsed_content.get("bid", "Pass")
            action_idx = self._BID_TO_INDEX.get(bid_str, 0)

            # Validity Check
            if not state.legal_action_mask[action_idx]:
                logger.warning(f"LLM suggested illegal bid {bid_str}. Fallback to Pass.")
                # We log this failure specifically
                DiagnosticLogger.log_turn(state, messages, content, f"ILLEGAL: {bid_str} -> Pass")
                action_idx = 0
            else:
                DiagnosticLogger.log_turn(state, messages, content, bid_str)

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            DiagnosticLogger.log_turn(state, messages, f"ERROR: {str(e)}", "Pass")
            action_idx = 0

        try:
            # 1. Initialize Error Dict
            errors = {
                'SV': 0, 'SC': 0, 'IA': 0, 
                'OG': 0, 'MG': 0, 'FH': 0, 
                'ID': 0, 'VB': 0, 'MSN': 0
            }

            # 2. Mathematical Checks (HCP/Fit)
            # You need the hand data. state.deal should exist if passed correctly.
            my_hcp, _, _ = self._get_hand_stats(state.observation)
            
            # Example: Check Overbidding Game (OG)
            # If bid level >= 4 and Combined HCP < 24
            # (You'll need to parse partner's HCP from previous reasoning or assume average)
            if action_idx >= 18: # 4C or higher
                # strict check: if my_hcp < 10 (assuming partner has ~12 max for non-forcing)
                if my_hcp < 8: # Conservative threshold example
                    errors['OG'] = 1

            # 3. Illegal Action (IA)
            if state.legal_action_mask[action_idx] == 0:
                errors['IA'] = 1

            # 4. Logic Parsing (SV, SC, etc.)
            # You must perform regex on 'reasoning_text' here.
            # Example:
            reasoning_text = parsed_content["reasoning"].lower() if "reasoning" in parsed_content else ""
            if "no fit" in reasoning_text and action_idx in [5,6,10,11]: # Bidding major
                errors['SV'] = 1

            # 1. SE - Over-Aggressive
            # Criteria: Bidding Game (Level 4+ or 3NT) with Combined HCP < 21 (User Spec)
            # We try to parse "combined_min_max" from the LLM's own JSON, or estimate it.
            combined_hcp_est = 0
            if "combined_min_max" in parsed_content:
                try:
                    # Parse "20-22" -> 21
                    rangestr = parsed_content["combined_min_max"].replace("HCP", "").strip()
                    low = int(rangestr.split("-")[0])
                    combined_hcp_est = low
                except:
                    combined_hcp_est = 25 # Benefit of doubt
            
            is_game_bid = (action_idx == 17) or (action_idx >= 18) # 3NT or 4C+
            if is_game_bid and combined_hcp_est < 21 and combined_hcp_est > 0:
                 errors['SE_Aggressive'] = 1

            # 2. SE - Wrong Strain
            # Criteria: Bidding NT (1NT, 2NT, 3NT...) when reasoning mentions a "Major fit"
            nt_bids = [7, 12, 17, 22, 27, 32, 37]
            if action_idx in nt_bids:
                if "major fit" in reasoning_text or "fit in heart" in reasoning_text or "fit in spade" in reasoning_text:
                    # Valid exception: If reasoning says "Major fit but stoppers..." (Context is hard)
                    # For now, strict check:
                    errors['SE_WrongStrain'] = 1

            # 3. SE - Passive
            # Criteria: Passing when reasoning admits a "Fit" and Opponents are bidding
            # Check if opponents have bid (look at bidding_history, non-zero values)
            opponents_active = np.any(state._bidding_history > 0) 
            if action_idx == 0: # Pass
                # Look for triggers in reasoning
                has_fit_reasoning = "10-card fit" in reasoning_text or "9-card fit" in reasoning_text or "strong fit" in reasoning_text
                if has_fit_reasoning and opponents_active:
                    errors['SE_Passive'] = 1

            # 5. Record to Tracker
            board_id = hash(state.observation.tobytes()) 
            lps_tracker.record_hand(board_id, errors)

        except Exception as e:
            # Don't let tracking crash the bidding
            print(f"LPS Tracking Error: {e}")
        # --- LPS TRACKING END ---
        
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