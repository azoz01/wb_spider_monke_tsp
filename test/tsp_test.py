import numpy as np

from engine.tsp import TspWrapper

SAMPLE_PROBLEM_PATH = "test/resources/rbg4.atsp"


def test_tsp_wrapper_initializes_without_error():
    TspWrapper.from_atsp_full_matrix(SAMPLE_PROBLEM_PATH)


def test_get_vertices():
    problem = TspWrapper.from_atsp_full_matrix(SAMPLE_PROBLEM_PATH)

    assert problem.get_vertices().tolist() == [0, 1, 2, 3]


def test_calculate_cost():
    problem = TspWrapper.from_atsp_full_matrix(SAMPLE_PROBLEM_PATH)
    path = np.array([3, 1, 0, 2])

    actual_cost = problem.calculate_cost(path)
    expected_cost = -22

    assert expected_cost == actual_cost


def test_get_edge_weight():
    problem = TspWrapper.from_atsp_full_matrix(SAMPLE_PROBLEM_PATH)

    assert problem._get_edge_weight(2, 3) == 12
