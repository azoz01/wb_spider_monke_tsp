from typing import Callable
import numpy as np

from itertools import starmap

from .permutation import PermutationEngine

np.random.seed(1)


def update_group_with_local_leader(
    group: np.ndarray,
    local_leader_idx_in_group: int,
    perturbation_rate: float,
    calculate_cost_callback: Callable[[np.ndarray], float],
) -> np.ndarray:

    (
        to_update_idx,
        potential_new_monkes,
    ) = __generate_new_potential_monkes_with_local_leader(
        group, local_leader_idx_in_group, perturbation_rate
    )

    group = __substitute_with_better_monkes(
        group, to_update_idx, potential_new_monkes, calculate_cost_callback
    )

    return group


def __generate_new_potential_monkes_with_local_leader(
    group: np.ndarray,
    local_leader_idx_in_group: int,
    perturbation_rate: float,
) -> tuple[np.ndarray, np.ndarray[np.ndarray]]:
    group = group.copy()
    local_leader = group[local_leader_idx_in_group]
    to_update_idx = np.random.random(size=group.shape[0]) >= perturbation_rate
    random_selected_monkes_idx = np.random.choice(
        np.arange(group.shape[0]), size=sum(to_update_idx)
    )
    random_selected_monkes = group[random_selected_monkes_idx]
    update_permutations = np.array(
        list(
            starmap(
                lambda monke, random_monke: PermutationEngine.compose_permutations(
                    PermutationEngine.permutation_difference(
                        monke, local_leader
                    ),
                    PermutationEngine.permutation_difference(
                        monke, random_monke
                    ),
                ),
                zip(group[to_update_idx], random_selected_monkes),
            ),
        )
    )
    potential_new_monkes = np.array(
        list(
            starmap(
                PermutationEngine.apply_permutation,
                zip(group[to_update_idx], update_permutations),
            )
        )
    )

    return to_update_idx, potential_new_monkes


def update_group_with_global_leader():
    pass


def __generate_new_potential_monkes_with_global_leader(
    group: np.ndarray,
    global_leader: np.ndarray,
    perturbation_rate: float,
    calculate_cost_callback: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray[np.ndarray]]:
    group = group.copy()
    global_leader_cost = calculate_cost_callback(global_leader, positive=True)
    probas = np.array(
        list(
            map(
                lambda monke: 0.1
                + 0.9
                * global_leader_cost
                / calculate_cost_callback(monke, positive=True),
                group,
            )
        )
    )
    to_update_idx = np.random.random(size=group.shape[0]) <= probas

    random_selected_monkes_idx = np.random.choice(
        np.arange(group.shape[0]), size=sum(to_update_idx)
    )
    random_selected_monkes = group[random_selected_monkes_idx]
    update_permutations = np.array(
        list(
            starmap(
                lambda monke, random_monke: PermutationEngine.compose_permutations(
                    PermutationEngine.permutation_difference(
                        monke, local_leader
                    ),
                    PermutationEngine.permutation_difference(
                        monke, random_monke
                    ),
                ),
                zip(group[to_update_idx], random_selected_monkes),
            ),
        )
    )
    potential_new_monkes = np.array(
        list(
            starmap(
                PermutationEngine.apply_permutation,
                zip(group[to_update_idx], update_permutations),
            )
        )
    )


def __substitute_with_better_monkes(
    group: np.ndarray,
    to_update_idx: np.ndarray,
    potential_new_monkes: np.ndarray,
    calculate_cost_callback: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    group = group.copy()
    potential_new_monkes_cost = np.array(
        list(map(calculate_cost_callback, potential_new_monkes))
    )
    actual_monkes_to_change_cost = np.array(
        list(map(calculate_cost_callback, group[to_update_idx]))
    )

    better_new_monke_idx = (
        potential_new_monkes_cost > actual_monkes_to_change_cost
    )
    to_update_idx[to_update_idx] &= better_new_monke_idx
    group[to_update_idx] = potential_new_monkes[better_new_monke_idx]

    return group
