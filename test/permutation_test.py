import numpy as np

from engine.permutation import PermutationEngine


def test_permutation_difference():
    perm_1 = np.array([1, 5, 0, 3, 2, 4])
    perm_2 = np.array([2, 3, 1, 5, 4, 0])

    permutation_difference = PermutationEngine.permutation_difference(
        perm_1, perm_2
    )
    actual_permutation = perm_1[permutation_difference]
    expected_permutation = perm_2

    assert permutation_difference.tolist() == [4, 3, 0, 1, 5, 2]
    assert actual_permutation.tolist() == expected_permutation.tolist()


def test_compose_permutations():
    perm_1 = np.array([1, 5, 0, 3, 2, 4])
    perm_2 = np.array([2, 3, 1, 5, 4, 0])

    actual_permutation = PermutationEngine.compose_permutations(perm_1, perm_2)
    expected_permutation = np.array([0, 3, 5, 4, 2, 1])

    assert expected_permutation.tolist() == actual_permutation.tolist()


def test_apply_permutation():
    perm_1 = np.array([1, 5, 0, 3, 2, 4])
    perm_2 = np.array([5, 4, 3, 2, 1, 0])

    actual_permutation = PermutationEngine.apply_permutation(perm_1, perm_2)
    expected_permutation = np.array([4, 2, 3, 0, 5, 1])

    assert expected_permutation.tolist() == actual_permutation.tolist()
