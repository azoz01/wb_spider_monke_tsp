import numpy as np


class PermutationEngine:
    @staticmethod
    def permutation_difference(
        perm_1: np.ndarray, perm_2: np.ndarray
    ) -> callable:
        arg_1 = np.argsort(perm_1)
        return arg_1[perm_2]

    @staticmethod
    def compose_permutations(
        perm_1: np.ndarray, perm_2: np.ndarray
    ) -> callable:
        return perm_1[perm_2]

    @staticmethod
    def apply_permutation(
        perm_1: np.ndarray, perm_2: np.ndarray
    ) -> np.ndarray:
        return perm_1[perm_2]
