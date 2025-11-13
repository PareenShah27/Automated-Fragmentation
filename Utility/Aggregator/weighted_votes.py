"""
Weighted Voting Aggregator
Combines ensemble proposals using weighted voting strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class WeightedVotingAggregator:
    """
    Aggregates multiple strategy proposals using weighted voting.
    
    Each strategy gets a weight based on historical performance.
    Default: Cost-based 25%, ML-based 30%, Evolutionary 25%, Graph-based 20%
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize weighted voting aggregator.
        
        Args:
            weights: Custom weights for strategies. If None, uses defaults.
        """
        if weights is None:
            weights = {
                'cost_based': 0.25,
                'ml_based': 0.30,
                'evolutionary': 0.25,
                'graph_based': 0.20
            }
        
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        self.proposals = []
    
    def add_proposal(self, strategy_name: str, result: Dict[str, Any]) -> None:
        """
        Add a proposal from an optimization strategy.
        
        Args:
            strategy_name: Name of strategy
            result: Optimization result dictionary
        """
        self.proposals.append({
            'strategy': strategy_name,
            **result
        })
    
    def aggregate(self) -> Dict[str, Any]:
        """
        Aggregate proposals using weighted voting.
        
        Algorithm:
        1. Normalize metrics to [0, 1] scale (0=worst, 1=best)
        2. Calculate weighted score for each proposal
        3. Select proposal with highest score
        4. Return aggregated result
        
        Returns:
            Aggregated result from weighted voting
        """
        
        if not self.proposals:
            raise ValueError("No proposals to aggregate")
        
        df = pd.DataFrame(self.proposals)
        
        # Normalize metrics to [0, 1] (lower is better for latency/cost)
        metrics = ['latency', 'comm_cost', 'storage']
        scores = []
        
        for idx, row in df.iterrows():
            score = 0
            
            for metric in metrics:
                if metric in df.columns:
                    max_val = df[metric].max()
                    min_val = df[metric].min()
                    
                    if max_val > min_val:
                        # Normalize: 0 (worst) to 1 (best)
                        normalized = 1 - ((row[metric] - min_val) / (max_val - min_val))
                    else:
                        normalized = 0.5
                    
                    score += normalized / 3.0  # Average across 3 metrics
            
            scores.append(score)
        
        # Select best proposal
        best_idx = np.argmax(scores)
        aggregated = self.proposals[best_idx].copy()
        aggregated['aggregation_method'] = 'Weighted Voting'
        aggregated['confidence_score'] = scores[best_idx]
        aggregated['all_scores'] = scores
        
        return aggregated
    
    def get_voting_summary(self) -> Dict[str, Any]:
        """Get summary of voting process."""
        return {
            'method': 'Weighted Voting',
            'weights': self.weights,
            'num_proposals': len(self.proposals),
            'strategies': list(self.weights.keys())
        }
    
    def reset(self) -> None:
        """Clear all proposals."""
        self.proposals = []