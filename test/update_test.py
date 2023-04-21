import numpy as np

from engine.update import (
    __generate_new_potential_monkes_with_local_leader,
    __substitute_with_better_monkes,
)


# TODO: przestudiowac bardziej ten kejs bo ja juz jebla dostaje
def test__generate_new_potential_monkes_with_local_leader():

    group = np.array([[4, 0, 3, 1, 2], [4, 1, 2, 3, 0], [4, 3, 1, 2, 0]])
    local_leader_idx_in_group = 2
    perturbation_rate = 0.2

    (
        actual_to_update_idx,
        actual_potential_new_monkes,
    ) = __generate_new_potential_monkes_with_local_leader(
        group,
        local_leader_idx_in_group,
        perturbation_rate,
    )

    expected_to_update_idx = np.array([True, True, False])
    expected_potential_new_monkes = np.array(
        [[4, 2, 0, 1, 3], [4, 0, 2, 3, 1]]
    )

    assert expected_to_update_idx.tolist() == actual_to_update_idx.tolist()
    assert (
        expected_potential_new_monkes.tolist()
        == actual_potential_new_monkes.tolist()
    )


def test__substitute_with_better_monkes():

    group = np.array([[4, 0, 3, 1, 2], [4, 1, 2, 3, 0], [4, 3, 1, 2, 0]])
    to_update_idx = np.array([True, True, False])
    potential_new_monkes = np.array([[4, 2, 0, 1, 3], [4, 0, 2, 3, 1]])

    def calculate_cost(x):
        x = x.tolist()
        if x == [4, 0, 3, 1, 2]:
            return -1
        elif x == [4, 1, 2, 3, 0]:
            return -2
        elif x == [4, 3, 1, 2, 0]:
            return -3
        elif x == [4, 2, 0, 1, 3]:
            return -2
        elif x == [4, 0, 2, 3, 1]:
            return -1

    actual_new_group = __substitute_with_better_monkes(
        group, to_update_idx, potential_new_monkes, calculate_cost
    )
    expected_new_group = np.array(
        [[4, 0, 3, 1, 2], [4, 0, 2, 3, 1], [4, 3, 1, 2, 0]]
    )

    assert actual_new_group.tolist() == expected_new_group.tolist()
