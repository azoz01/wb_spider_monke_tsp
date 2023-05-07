from __future__ import annotations

import numpy as np

from operator import attrgetter
from time import time
from tqdm import tqdm
from typing import List, Tuple

from engine.tsp import TspWrapper
from engine.report import make_report
from engine.utils import (
    decompose_permutation_to_swap_sequence,
    permutation_difference,
    choose_in_order,
    merge_swap_sequences,
    apply_swap_sequence,
)


class MonkeOptimizer:
    def __init__(
        self,
        allowed_maximum_groups,
        perturbation_rate,
        local_leader_limit,
        global_leader_limit,
        monkes_count,
    ):
        self.allowed_maximum_groups = allowed_maximum_groups
        self.perturbation_rate = perturbation_rate
        self.local_leader_limit = local_leader_limit
        self.global_leader_limit = global_leader_limit
        self.monkes_count = monkes_count
        self.global_leader_count = 0

    def optimize(
        self,
        tsp_problem: TspWrapper,
        run_name: str,
        timeout_seconds: int = 3600,
        n_iter: int = 1000,
    ) -> Tuple[np.ndarray, float]:
        results = {
            "iteration": [],
            "best_cost": [],
            "number_of_groups": [],
            "iteration_time": [],
            "local_leader_phase_time": [],
            "global_leader_phase_time": [],
            "local_leader_learning": [],
            "global_leader_learning": [],
            "local_leader_decision_phase": [],
            "global_leader_decision_phase": [],
        }

        start_time = time()
        vertices = tsp_problem.get_vertices()
        monkes = [np.random.permutation(vertices) for _ in range(self.monkes_count)]

        costs = list(map(tsp_problem.calculate_cost, monkes))
        self.global_leader = monkes[np.argmin(costs)]
        groups = [self.MonkeGroup(monkes, self.global_leader)]

        iterator = tqdm(range(n_iter))
        best_solution = self.global_leader
        best_cost = tsp_problem.calculate_cost(self.global_leader)
        for i in iterator:
            iteration_start_time = time()

            time_start_local_leader_phase = time()
            for group in groups:
                self.local_leader_phase(group=group, tsp_problem=tsp_problem)
            time_end_local_leader_phase = time()

            time_start_global_leader_phase = time()
            for group in groups:
                self.global_leader_phase(
                    group=group,
                    global_leader=self.global_leader,
                    tsp_problem=tsp_problem,
                )
            time_end_global_leader_phase = time()

            time_start_local_leader_learning = time()
            self.local_leader_learning(groups=groups, tsp_problem=tsp_problem)
            time_end_local_leader_learning = time()

            time_start_global_leader_learning = time()
            self.global_leader_learning(groups=groups, tsp_problem=tsp_problem)
            time_end_global_leader_learning = time()

            time_start_local_leader_decision_phase = time()
            self.local_leader_decision_phase(
                groups=groups, vertices=vertices, tsp_problem=tsp_problem
            )
            time_end_local_leader_decision_phase = time()

            time_start_global_leader_decision_phase = time()
            groups = self.global_leader_decision_phase(
                groups=groups, tsp_problem=tsp_problem
            )
            time_end_global_leader_decision_phase = time()

            global_leader_cost = tsp_problem.calculate_cost(self.global_leader)
            if global_leader_cost < best_cost:
                best_solution = self.global_leader.copy()
                best_cost = global_leader_cost

            iterator.set_description(
                f"Lowest cost found: {best_cost}, number of groups: {len(groups)}"
            )

            iteration_end_time = time()

            # update report dict
            results["iteration"].append(i)
            results["best_cost"].append(best_cost)
            results["number_of_groups"].append(len(groups))
            results["iteration_time"].append(iteration_end_time - iteration_start_time)
            results["local_leader_phase_time"].append(
                time_end_local_leader_phase - time_start_local_leader_phase
            )
            results["global_leader_phase_time"].append(
                time_end_global_leader_phase - time_start_global_leader_phase
            )
            results["local_leader_learning"].append(
                time_end_local_leader_learning - time_start_local_leader_learning
            )
            results["global_leader_learning"].append(
                time_end_global_leader_learning - time_start_global_leader_learning
            )
            results["local_leader_decision_phase"].append(
                time_end_local_leader_decision_phase
                - time_start_local_leader_decision_phase
            )
            results["global_leader_decision_phase"].append(
                time_end_global_leader_decision_phase
                - time_start_global_leader_decision_phase
            )

            if time() - start_time > timeout_seconds:
                break

        run_params = {
            "allowed_maximum_groups": self.allowed_maximum_groups,
            "perturbation_rate": self.perturbation_rate,
            "local_leader_limit": self.local_leader_limit,
            "global_leader_limit": self.global_leader_limit,
            "monkes_count": self.monkes_count,
        }

        make_report(
            path=run_name,
            run_params=run_params,
            results_dict=results,
            run_time=start_time,
            best_solution=best_solution,
            best_cost=best_cost,
        )

        return best_solution, best_cost

    class MonkeGroup:
        def __init__(self, monkes: np.ndarray, leader: np.ndarray):
            self.monkes = monkes
            self.leader = leader
            self.counter = 0

        def __len__(self):
            return len(self.monkes)

    def local_leader_phase(self, group: MonkeGroup, tsp_problem: TspWrapper):
        for i in range(len(group)):
            if np.random.random() >= self.perturbation_rate:
                monke = group.monkes[i]
                random_monke = group.monkes[np.random.choice(np.arange(len(group)))]

                ss1 = decompose_permutation_to_swap_sequence(
                    permutation_difference(monke, group.leader)
                )
                ss1 = choose_in_order(ss1)
                ss2 = decompose_permutation_to_swap_sequence(
                    permutation_difference(monke, random_monke)
                )
                ss2 = choose_in_order(ss2)
                merged_ss = merge_swap_sequences(ss1, ss2)
                new_monke = apply_swap_sequence(monke, merged_ss)
                if tsp_problem.calculate_cost(new_monke) < tsp_problem.calculate_cost(
                    monke
                ):
                    group.monkes[i] = new_monke

    def global_leader_phase(
        self,
        group: MonkeGroup,
        global_leader: np.ndarray,
        tsp_problem: TspWrapper,
    ):
        for i in range(len(group)):
            monke = group.monkes[i]
            prob = (
                0.9
                * tsp_problem.calculate_cost(global_leader)
                / tsp_problem.calculate_cost(monke)
                + 0.1
            )
            if np.random.random() <= prob:
                random_monke = group.monkes[np.random.choice(np.arange(len(group)))]
                ss1 = decompose_permutation_to_swap_sequence(
                    permutation_difference(monke, global_leader)
                )
                ss1 = choose_in_order(ss1)
                ss2 = decompose_permutation_to_swap_sequence(
                    permutation_difference(monke, random_monke)
                )
                ss2 = choose_in_order(ss2)
                merged_ss = merge_swap_sequences(ss1, ss2)
                new_monke = apply_swap_sequence(monke, merged_ss)
                if tsp_problem.calculate_cost(new_monke) < tsp_problem.calculate_cost(
                    monke
                ):
                    group.monkes[i] = new_monke

    def local_leader_learning(self, groups, tsp_problem: TspWrapper):
        for group in groups:
            monkes_cost = list(map(tsp_problem.calculate_cost, group.monkes))
            leader_cost = tsp_problem.calculate_cost(group.leader)
            min_cost_monke_idx = np.argmin(monkes_cost)
            if monkes_cost[min_cost_monke_idx] < leader_cost:
                group.leader = group.monkes[min_cost_monke_idx]
                group.counter = 0
            else:
                group.counter += 1

    def global_leader_learning(self, groups: List[MonkeGroup], tsp_problem: TspWrapper):
        local_leaders = list(map(attrgetter("leader"), groups))
        leaders_costs = list(map(tsp_problem.calculate_cost, local_leaders))
        min_cost_leader_idx = np.argmin(leaders_costs)
        if leaders_costs[min_cost_leader_idx] < tsp_problem.calculate_cost(
            self.global_leader
        ):
            self.global_leader = groups[min_cost_leader_idx].leader
            self.global_leader_count = 0
        else:
            self.global_leader_count += 1

    def local_leader_decision_phase(
        self,
        groups: List[MonkeGroup],
        vertices: np.array,
        tsp_problem: TspWrapper,
    ):
        for group in groups:
            if group.counter > self.local_leader_limit:
                group.counter = 0
                if np.random.uniform() < self.perturbation_rate:
                    for i in range(len(group)):
                        monke = group.monkes[i]
                        ss1 = decompose_permutation_to_swap_sequence(
                            permutation_difference(monke, self.global_leader)
                        )
                        ss1 = choose_in_order(ss1)
                        ss2 = decompose_permutation_to_swap_sequence(
                            permutation_difference(monke, group.leader)
                        )
                        ss2 = choose_in_order(ss2)
                        new_monke = apply_swap_sequence(monke, ss1)
                        new_monke = apply_swap_sequence(new_monke, ss2)
                        group.monkes[i] = new_monke
                else:
                    group.monkes = [
                        np.random.permutation(vertices)
                        for _ in range(self.monkes_count)
                    ]
                best_monke_in_group_idx = np.argmin(
                    list(map(tsp_problem.calculate_cost, group.monkes))
                )
                group.leader = group.monkes[best_monke_in_group_idx]

    def global_leader_decision_phase(
        self, groups: MonkeGroup, tsp_problem: TspWrapper
    ) -> List[MonkeGroup]:
        if self.global_leader_count > self.global_leader_limit:
            self.global_leader_count = 0
            if len(groups) < self.allowed_maximum_groups:
                group = groups.pop(0)
                group_new_1 = group.monkes[0 : len(group) // 2]
                group_new_2 = group.monkes[len(group) // 2 + 1 :]
                group_new_1 = self.MonkeGroup(
                    group_new_1,
                    group_new_1[
                        np.argmin(list(map(tsp_problem.calculate_cost, group_new_1)))
                    ],
                )
                group_new_2 = self.MonkeGroup(
                    group_new_2,
                    group_new_2[
                        np.argmin(list(map(tsp_problem.calculate_cost, group_new_2)))
                    ],
                )
                groups.append(group_new_1)
                groups.append(group_new_2)
            else:
                consolidated_group_monkes = np.concatenate(
                    list(map(attrgetter("monkes"), groups))
                )
                leader = self.global_leader
                groups = [self.MonkeGroup(consolidated_group_monkes, leader)]
        return groups
