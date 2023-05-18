from __future__ import annotations

import numpy as np
import tsplib95

from functools import lru_cache

from engine.utils import numpy_to_tuple


class TspWrapper:
    def __init__(self, problem: tsplib95.models.Problem) -> None:
        self.problem = problem

    def get_vertices(self) -> list[int]:
        return np.array(list(self.problem.get_nodes()))

    @numpy_to_tuple
    @lru_cache(maxsize=10_000)
    def calculate_cost(self, path: np.ndarray) -> int:
        return np.sum(
            [self._get_edge_weight(v1, v2) for v1, v2 in zip(path[:-1], path[1:])]
        ) + self._get_edge_weight(path[-1], path[0])

    def _get_edge_weight(self, vertex_1: int, vertex_2: int) -> int:
        return self.problem.get_weight(vertex_1, vertex_2)

    @staticmethod
    def from_atsp_full_matrix(path: str):
        problem = tsplib95.load(path)
        return TspWrapper(problem)
