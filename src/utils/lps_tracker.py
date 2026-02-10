import numpy as np
import collections
import json
import os
import threading

class LPSTracker:
    _instance = None
    _lock = threading.Lock()
    LOG_FILE = "src/logs/lps_stats.jsonl" # Shared file for inter-process communication
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LPSTracker, cls).__new__(cls)
        return cls._instance

    def reset(self):
        """
        Called by run_bidding.py (Main Process) at the start.
        Clears the shared log file.
        """
        if os.path.exists(self.LOG_FILE):
            try:
                os.remove(self.LOG_FILE)
            except OSError:
                pass # File might be in use or already gone

    def record_hand(self, board_id, errors):
        """
        Called by llm.py (Server). Records Logic and Strategic errors.
        """
        # Weighted Scoring
        # Fatal (50): System Violations, Contradictions, Illegal Actions
        fatal = (errors.get('SV', 0) + errors.get('SC', 0) + errors.get('IA', 0)) * 50
        
        # Math (20): Overbids, Fit Hallucinations
        math = (errors.get('OG', 0) + errors.get('MG', 0) + errors.get('FH', 0)) * 20
        
        # Strategy (10): The new metrics (Passive, Aggressive, Strain)
        # Note: We weight these lower than Fatal errors, but high enough to hurt.
        strat = (errors.get('SE_Passive', 0) + 
                 errors.get('SE_Aggressive', 0) + 
                 errors.get('SE_WrongStrain', 0)) * 10
        
        total_lps = fatal + math + strat

        entry = {
            "type": "process",
            "board_id": board_id,
            "lps": total_lps,
            "details": errors 
        }

        self._write_entry(entry)

    def _write_entry(self, entry):
        with self._lock:
            with open(self.LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def record_outcome(self, imps):
        """
        Records the strategic outcome.
        - Negative IMPs (< -2): Recorded as Failures (Penalties)
        - Positive IMPs (> 5): Recorded as Successes (Efficiency Boosters)
        """
        entry = None
        if imps <= -2.0:
            entry = {"type": "outcome_fail", "imps": float(imps)}
        elif imps >= 5.0:
             # NEW: Track big wins
            entry = {"type": "outcome_win", "imps": float(imps)}
            
        if entry:
            self._write_entry(entry)

    def get_final_metrics(self):
        """
        Returns:
        1. LPS Score (Lower is better) - The "Compliance" score.
        2. Efficiency Ratio (Higher is better) - The "Skill" score.
        """
        if not os.path.exists(self.LOG_FILE):
            return 0.0, 0.0, 0, 0.0

        process_scores = []
        outcome_penalties = 0.0
        total_positive_imps = 0.0 # NEW
        decisions_made = 0

        try:
            with open(self.LOG_FILE, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if data["type"] == "process":
                            process_scores.append(data["lps"])
                            decisions_made += 1
                        elif data["type"] == "outcome_fail":
                            penalty = abs(data["imps"]) * 5
                            outcome_penalties += penalty
                        elif data["type"] == "outcome_win":
                            # NEW: Accumulate positive IMPs
                            total_positive_imps += data["imps"]
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return 0.0, 0.0, 0, 0.0

        if not process_scores:
            return 0.0, 0.0, 0, 0.0

        # 1. Calculate Compliance (LPS) - LOWER IS BETTER
        avg_process_lps = np.mean(process_scores)
        avg_outcome_penalty = outcome_penalties / max(1, decisions_made)
        final_lps_score = avg_process_lps + avg_outcome_penalty
        
        # 2. Calculate Efficiency - HIGHER IS BETTER
        # Formula: Total Wins / (Total Logic Errors + 1)
        # This prevents "Lucky Wins" (High errors) from inflating the score too much.
        total_logic_errors = sum(process_scores) # Raw sum of error points
        efficiency_score = total_positive_imps / (total_logic_errors + 1.0)
        
        return final_lps_score, efficiency_score, decisions_made, outcome_penalties

# Global instance
lps_tracker = LPSTracker()