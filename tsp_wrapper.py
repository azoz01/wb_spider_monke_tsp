from __future__ import annotations

import numpy as np
import tsplib95
from itertools import chain

class TspWrapper:

    def __init__(self, problem: tsplib95.models.Problem) -> None:
        self.problem = problem

    def get_edge_weight(self, vertex_1: int, vertex_2: int) -> int:
        return self.problem.get_weight(vertex_1, vertex_2)

    @staticmethod
    def from_atsp_full_matrix(path: str):
        problem = tsplib95.load(path)
        return TspWrapper(problem)