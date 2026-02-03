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
        TWO_ONE_AGGRESSIVE_STANDARD = (
            "You and your partner follow an aggressive 2/1 Game Forcing bidding system. "
            "Bids are commitments, not suggestions. Forcing bids must be responded to.",

            "Opening bids: 1 of a suit shows 12-21 HCP (5+ cards in majors, 3+ in minors). "
            "1NT opening shows 15-17 HCP and a balanced hand with no 5-card major.",

            "2/1 principle: A 2-level response to a 1-level suit opening (e.g. 1H-2C, 1S-2D) "
            "shows 12+ HCP, 4+ cards in the bid suit, and is game-forcing. "
            "After a 2/1 bid, neither partner may pass until game is reached or clearly ruled out.",

            "1NT responses to a major opening (1H-1NT or 1S-1NT) show 6-11 HCP, are not forcing, "
            "and deny 3+ card support for opener's major. This bid limits the hand.",

            "Raising partner's major: Single raise (to 2 of the major) shows 6-9 HCP and 3+ support. "
            "Limit raise (to 3 of the major) shows 10-12 HCP and 4+ support. "
            "Direct game raise (to 4 of the major) shows 13+ HCP or strong distribution and ends exploration.",

            "After a 2/1 response, opener rebids to describe shape first, strength second. "
            "Rebidding a suit shows extra length, bidding a new suit shows 4+ cards, "
            "and notrump rebids show balanced hands (2NT ≈ 18-19, 3NT ≈ 15-17 with stoppers).",

            "General aggression rules: With approximately 25+ combined HCP, bid game. "
            "Do not stop in partscore after a 2/1 sequence. Prefer showing shape with new suits "
            "over repeating notrump when unbalanced.",

            "Interpretation rules: New suits are natural. Failure to raise partner's suit implies lack of support. "
            "Passing is only allowed after explicitly non-forcing bids.",

            "If multiple interpretations are possible, assume partner is maximum for their bid, "
            "assume bids are natural, and choose the action that keeps the auction game-forcing."
        )
        # self.system_prompt = (
        #     "You are a World-Class Bridge Engine."
        #     + "\n".join(TWO_ONE_AGGRESSIVE_STANDARD)
        #     + "\n### CRITICAL CONTEXT-AWARE STRATEGY:\n"

        #     ### OPENING BID AUTHORITY (MANDATORY):
        #     "When you are the opening bidder (no prior bids in the auction):"
        #     "- You MUST NOT invent or derive an opening bid from first principles."
        #     "- You MUST choose from a restricted, predefined opening set."
        #     "- Treat opening as selecting the least exploitable signal, not as hand expression."
        #     "You are allowed to open ONLY with one of the following actions:"
        #     "- PASS"
        #     "- 1C"
        #     "- 1D"
        #     "- 1H"
        #     "- 1S"
        #     "- 1NT"
        #     "All other opening bids (2C, 2D, weak twos, preempts, jump openings) are DISALLOWED unless explicitly instructed elsewhere."
        #     "Your task when opening is:"
        #     "- Evaluate which allowed opening minimizes expected IMP loss against the baseline,"
        #     "- NOT to maximize descriptive accuracy."
            
        #     # 1. WHEN WE OPEN THE BIDDING (1st/2nd position)
        #     "**IF I AM THE OPENING SIDE (I or partner opened first):**\n"
        #     "A. OPENING SELECTION (EVALUATIVE, NOT CREATIVE):"
        #     "When choosing an opening bid from the allowed set:"
        #     "- Use HCP and shape ONLY to decide between PASS, 1NT, or 1 of a suit."
        #     "- Borderline hands (11-12 HCP) should bias toward PASS unless:"
        #     "- A 5-card major is present, OR"
        #     "- The hand is unbalanced (singleton or void)."
        #     "Do NOT downgrade openings due to fear of later commitment. Opening does not promise extras."
        #     "An opening bid is not a promise of strength beyond its minimum range. Subsequent actions, not the opening, define commitment."
            
        #     "B. RESPONSE TO COMPETITION (DEFENSIVE):\n"
        #     "   - If baseline overcalls: They are often light. DOUBLE with:\n"
        #     "     * 4+ cards in their suit with A/K\n"
        #     "     * 13+ HCP total\n"
        #     "   - Do NOT stretch to show a second suit without 15+ HCP\n"
        #     "   - After our opening and baseline interference: Assume partner is minimum unless they bid again voluntarily. A voluntary second action by partner implies extra values or shape.\n"
            
        #     "C. GAME BIDDING (CAUTIOUS):\n"
        #     "   - After our opening, require 26+ combined HCP for game\n"
        #     "   - Prefer 3NT over 5m unless exceptional fit\n"
        #     "   - Stop in 2M when combined HCP < 24\n"
            
        #     # 2. WHEN BASELINE OPENS FIRST (OUR STRENGTH)
        #     "**IF OPPONENTS (BASELINE) OPEN FIRST:**\n"
        #     "A. OVERCALL AGGRESSION (MAXIMUM):\n"
        #     "   - Overcall at 1-level with: 8+ HCP, 5+ card suit\n"
        #     "   - Overcall at 2-level with: 10+ HCP, good 5+ suit\n"
        #     "   - Jump overcall (2M over 1m): 15-17 HCP, 6+ good suit\n"
            
        #     "B. PENALTY DOUBLES (EXPLOITATIVE):\n"
        #     "   - DOUBLE baseline's 1NT (15-17) with: 15+ HCP balanced\n"
        #     "   - DOUBLE their suit bids with: 4+ trumps including 2 honors\n"
        #     "   - Baseline overbids frequently—punish it\n"
            
        #     "C. PART-SCORE BATTLES (DOMINATE):\n"
        #     "   - Compete to 3-level with 9+ card fit\n"
        #     "   - Sacrifice at 4-level if non-vulnerable vs vulnerable\n"
            
        #     # 3. VULNERABILITY ADJUSTMENTS
        #     "**VULNERABILITY SPECIFICS:**\n"
        #     "   - NON-VUL (White) when opening: Conservative preempts, avoid -200\n"
        #     "   - VUL (Red) when opening: Sound openings, bid thin games (40%)\n"
        #     "   - NON-VUL when defending: Aggressive sacrifices, push them\n"
        #     "   - VUL when defending: Take safe penalties, bid solid games\n"
            
        #     # 4. PARTNERSHIP LOGIC
        #     "**PARTNERSHIP INFERENCE:**\n"
        #     "   - When partner passes initially: They have 0-5 HCP\n"
        #     "   - When partner responds 1NT: They have 6-10 HCP, no fit\n"
        #     "   - When partner makes a 2/1: They have 12+ HCP—force to game\n"

        #     "\nOUTPUT FORMAT (JSON ONLY):\n"
        #     "{{\n"
        #     "  'context': 'Opening side or Defending side?',\n"
        #     "  'hand_evaluation': 'HCP, Quick Tricks, Shape, Seat, Vulnerability',\n"
        #     "  'system_state': 'Auction interpretation with HCP ranges',\n"
        #     "  'strategy_chosen': 'Which specific strategy above applies?',\n"
        #     "  'mental_simulation': 'If I bid X, likely outcomes given context',\n"
        #     "  'bid': 'EXACT_BID_STRING'\n"
        #     "}}"
        # )
        self.system_prompt = (
            "You are an expert bridge bidding agent. Before each bid, you must complete a structured analysis.\n"
            
            "### REQUIRED ANALYSIS STEPS:\n"
            
            "**STEP 1: HAND EVALUATION**\n"
            "- Count HCP (A=4, K=3, Q=2, J=1)\n"
            "- Count distribution points (5-card suit +1, 6-card +2, etc.)\n"
            "- Identify your longest suit(s)\n"
            "- Note vulnerability (Vul/Non-vul)\n"
            
            "**STEP 2: AUCTION ANALYSIS**\n"
            "- What has partner shown? (HCP range and shape)\n"
            "- What have opponents shown? (HCP range and shape)\n"
            "- Is this a forcing auction or can I pass?\n"
            "- What is partner's likely HCP range?\n"
            
            "**STEP 3: PARTNERSHIP STRENGTH**\n"
            "- My HCP: X\n"
            "- Partner's likely range: Y to Z HCP\n"
            "- Combined minimum: X + Y\n"
            "- Combined maximum: X + Z\n"
            
            "**STEP 4: CONTRACT TARGET**\n"
            "- If combined 25+ HCP → Bid game (3NT/4M/5m)\n"
            "- If combined 23-24 HCP → Invite game (2NT/3M)\n"
            "- If combined <23 HCP → Stop in part-score\n"
            "- If combined 33+ HCP → Consider slam (BUT ONLY IF PARTNER SHOWED EXTRAS)\n"
            
            "**STEP 5: BID SELECTION**\n"
            "- Available bids that match target level\n"
            "- Choose bid that best describes hand\n"
            "- Prefer major suits over minors for game\n"
            "- Prefer 3NT over 5m unless 9+ card fit\n"
            
            "**STEP 6: SAFETY CHECK**\n"
            "- Am I overbidding? (combined HCP too low for this level?)\n"
            "- Am I underbidding? (missing game with 25+ combined?)\n"
            "- Could this go down badly? (going to 5-level on 23 HCP?)\n"
            
            "\nOUTPUT FORMAT (JSON - COMPLETE ALL FIELDS):\n"
            "{{\n"
            "  'my_hcp': 'Number',\n"
            "  'partner_range': 'X-Y HCP based on bids',\n"
            "  'combined_min_max': 'Min-Max combined HCP',\n"
            "  'target_level': 'Part-score/Invite/Game/Slam',\n"
            "  'reasoning': 'Why this bid is correct',\n"
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
        
        # try:
        #     completion = self.client.chat.completions.create(
        #         model=self.model_name,
        #         messages=messages,
        #         temperature=0.0,  # Use deterministic output for testing
        #         response_format={"type": "json_object"} # Enforce JSON output
        #     )
            
        #     # 1. Parse the JSON response
        #     raw_content = completion.choices[0].message.content
        #     try:
        #         response_data = json.loads(raw_content)
        #         reasoning = response_data.get("imp_rationale", "No reasoning provided.")
        #         bid = response_data.get("bid", "Pass") # Default to Pass if key missing
                
        #         # 2. Print the reasoning and bid for debugging
        #         print(f"\n[LLM Reasoning]: {reasoning}")
        #         print(f"[LLM Bid]: {bid}\n")
                
        #     except json.JSONDecodeError:
        #         logger.error(f"LLM returned invalid JSON: {raw_content}. Falling back to 'Pass'.")
        #         bid = "Pass"

        #     # 3. Clean up whitespace/quotes (Standard Logic)
        #     bid = str(bid).replace('"', '').replace("'", '').strip()
            
        #     # 4. Check if the bid is valid and legal (Standard Logic)
        #     if bid not in self._BID_TO_INDEX:
        #         logger.error(f"LLM returned unknown bid: '{bid}'. Falling back to 'Pass'.")
        #         action_idx = self._BID_TO_INDEX["Pass"]
        #     else:
        #         action_idx = self._BID_TO_INDEX[bid]

        #         # Ensure the chosen action is actually legal
        #         if not state.legal_action_mask[action_idx]:
        #             logger.error(f"LLM chose illegal bid '{bid}'. Legal options: {self._format_bidding_history(state.legal_action_mask)}. Falling back to 'Pass'.")
        #             action_idx = self._BID_TO_INDEX["Pass"]

        # except Exception as e:
        #     logger.error(f"OpenAI API call failed: {e}. Returning 'Pass' as fallback.")
        #     action_idx = self._BID_TO_INDEX["Pass"]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # --- DIAGNOSTIC LOGGING ---
            # We log BEFORE we return, so we capture the raw thought process
            parsed_content = json.loads(content)
            bid_str = parsed_content.get("bid", "Pass")
            
            # Map string back to index
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