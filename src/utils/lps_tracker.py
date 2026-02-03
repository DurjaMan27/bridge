import numpy as np
import collections

class LPSTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LPSTracker, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.board_records = collections.defaultdict(list)
        
    def record_hand(self, board_id, errors):
        fatal = (errors.get('SV', 0) + errors.get('SC', 0) + errors.get('IA', 0)) * 50
        math = (errors.get('OG', 0) + errors.get('MG', 0) + errors.get('FH', 0)) * 20
        strat = (errors.get('ID', 0) + errors.get('VB', 0) + errors.get('MSN', 0)) * 5
        
        total_lps = fatal + math + strat
        self.board_records[board_id].append(total_lps)

    def get_final_metrics(self):
        consistency_scores = []
        for board_id, scores in self.board_records.items():
            if not scores: continue
            # If running only once, std is 0.0
            avg_lps = np.mean(scores)
            std_lps = np.std(scores) if len(scores) > 1 else 0.0
            consistency_scores.append(avg_lps + std_lps)
            
        if not consistency_scores:
            return 0.0, 0
            
        return np.mean(consistency_scores), len(consistency_scores)

# Global instance
lps_tracker = LPSTracker()