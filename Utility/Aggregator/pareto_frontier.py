"""
Pareto Frontier Aggregator
Computes and selects solutions from Pareto frontier for multi-objective optimization.
"""

import numpy as np
from typing import Dict, List, Any, Optional


class ParetoFrontierAggregator:
    """
    Computes Pareto optimal frontier from ensemble proposals.
    
    Pareto Optimality: A solution is on the frontier if no other solution
    is better in all objectives simultaneously.
    
    Useful for multi-objective optimization with tradeoffs.
    """
    
    def __init__(self):
        """Initialize Pareto frontier aggregator."""
        self.proposals = []
        self.frontier = []
        self.preferences = None
    
    def add_proposal(self, strategy_name: str, result: Dict[str, Any]) -> None:
        """
        Add a proposal from optimization strategy.
        
        Args:
            strategy_name: Name of strategy
            result: Optimization result
        """
        self.proposals.append({
            'strategy': strategy_name,
            **result
        })
    
    def compute_frontier(self) -> List[Dict[str, Any]]:
        """
        Compute Pareto frontier from all proposals.
        
        Algorithm:
        1. For each proposal P:
           - Check if dominated by any proposal in frontier
           - If not dominated, add to frontier
           - Remove any frontier solutions dominated by P
        
        Returns:
            List of non-dominated (Pareto optimal) solutions
        """
        
        if not self.proposals:
            return []
        
        frontier = []
        
        for prop in self.proposals:
            dominated = False
            
            # Check if prop is dominated by any solution in frontier
            for f_prop in frontier:
                if self._dominates(f_prop, prop):
                    dominated = True
                    break
            
            if not dominated:
                # Remove frontier solutions that are now dominated by prop
                frontier = [f for f in frontier if not self._dominates(prop, f)]
                frontier.append(prop)
        
        self.frontier = frontier
        return frontier
    
    def _dominates(self, solution1: Dict, solution2: Dict) -> bool:
        """
        Check if solution1 strictly dominates solution2.
        
        Solution1 dominates Solution2 if:
        - solution1 is better or equal in ALL objectives
        - solution1 is strictly better in AT LEAST ONE objective
        
        For fragmentation (lower is better for latency/cost):
        
        Args:
            solution1: First solution
            solution2: Second solution
        
        Returns:
            True if solution1 dominates solution2
        """
        
        metrics = ['latency', 'comm_cost', 'storage', 'balance']
        
        better_or_equal = 0
        strictly_better = 0
        
        for metric in metrics:
            if metric in solution1 and metric in solution2:
                val1 = solution1[metric]
                val2 = solution2[metric]
                
                # For balance, higher is better (inverted)
                if metric == 'balance':
                    if val1 >= val2:
                        better_or_equal += 1
                        if val1 > val2:
                            strictly_better += 1
                else:  # lower is better for latency, cost, storage
                    if val1 <= val2:
                        better_or_equal += 1
                        if val1 < val2:
                            strictly_better += 1
        
        # Dominates if better/equal in all AND strictly better in >= 1
        return better_or_equal >= len(metrics) and strictly_better > 0
    
    def select_solution(self, preferences: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Select best solution from Pareto frontier based on user preferences.
        
        Args:
            preferences: Weight dict {'latency': 0.4, 'comm_cost': 0.3, ...}
        
        Returns:
            Selected solution from frontier
        """
        
        if not self.frontier:
            raise ValueError("No frontier to select from. Compute frontier first.")
        
        if preferences is None:
            # Default: equal weight
            preferences = {
                'latency': 0.25,
                'comm_cost': 0.25,
                'storage': 0.25,
                'balance': 0.25
            }
        
        self.preferences = preferences
        
        # Score each frontier solution
        scores = []
        for solution in self.frontier:
            score = 0
            for metric, weight in preferences.items():
                if metric in solution:
                    # Normalize metric (0-1 scale, 1=best)
                    val = solution[metric]
                    if metric == 'balance':
                        # Higher is better
                        normalized = min(1.0, val / 0.5)  # Assume max 0.5
                    else:
                        # Lower is better
                        normalized = max(0.0, 1.0 - val / 100)
                    
                    score += weight * normalized
            scores.append(score)
        
        best_idx = np.argmax(scores)
        selected = self.frontier[best_idx].copy()
        selected['selection_method'] = 'Pareto + Preference'
        selected['preference_score'] = scores[best_idx]
        
        return selected
    
    def get_frontier_summary(self) -> Dict[str, Any]:
        """Get summary of Pareto frontier."""
        
        if not self.frontier:
            return {'frontier_size': 0}
        
        return {
            'frontier_size': len(self.frontier),
            'num_proposals': len(self.proposals),
            'frontier_diversity': self._compute_diversity(),
            'method': 'Pareto Frontier Selection'
        }
    
    def _compute_diversity(self) -> float:
        """
        Compute diversity score of frontier (0-1).
        
        Higher = more diverse tradeoffs available
        
        Returns:
            Diversity score
        """
        
        if len(self.frontier) < 2:
            return 0.0
        
        # Calculate metric range across frontier
        metrics = ['latency', 'comm_cost', 'storage']
        total_range = 0
        
        for metric in metrics:
            values = [s.get(metric, 0) for s in self.frontier]
            metric_range = max(values) - min(values)
            total_range += metric_range
        
        # Normalize diversity (higher = more diverse)
        return min(1.0, total_range / 300)  # Arbitrary normalization
    
    def reset(self) -> None:
        """Clear all proposals and frontier."""
        self.proposals = []
        self.frontier = []
        self.preferences = None