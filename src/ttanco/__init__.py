"""Verification-first test-time adaptation for neural combinatorial optimization."""

from ttanco.adaptation import SearchConfig, SearchResult, run_method
from ttanco.domain import TSPInstance, TourSolution, solve_held_karp
from ttanco.model import EdgePolicy, PolicyConfig

__all__ = [
    "EdgePolicy",
    "PolicyConfig",
    "SearchConfig",
    "SearchResult",
    "TSPInstance",
    "TourSolution",
    "run_method",
    "solve_held_karp",
]

__version__ = "0.1.0"
