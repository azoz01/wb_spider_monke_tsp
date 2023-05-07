import numpy as np


def permutation_difference(perm_1: np.ndarray, perm_2: np.ndarray) -> callable:
    arg_1 = np.argsort(perm_1)
    return arg_1[perm_2]


def decompose_permutation_to_swap_sequence(permutation):
    swaps = []
    n = len(permutation)
    vistited = [0] * n

    for i in range(n):
        if vistited[i] == 0:
            vistited[i] = 1
            if permutation[i] != i:
                curr_idx = i
                while vistited[permutation[curr_idx]] == 0:
                    swaps.append((curr_idx, permutation[curr_idx]))
                    curr_idx = permutation[curr_idx]
                    vistited[curr_idx] = 1

    return np.array(swaps)


def swap_sequence_to_permutation(ss):
    permutation = np.arange(np.max(ss) + 1)
    for idx1, idx2 in ss:
        permutation[idx1], permutation[idx2] = permutation[idx2], permutation[idx1]
    return permutation


def merge_swap_sequences(ss1, ss2):
    return ss1.tolist() + ss2.tolist()


def apply_swap_sequence(perm, ss):
    perm = perm.copy()
    for so in ss:
        perm[so[1]], perm[so[0]] = perm[so[0]], perm[so[1]]
    return perm


def choose_in_order(l):
    if len(l) == 0:
        return np.array([])
    length = np.random.choice(np.arange(1, len(l) + 1))
    idx = np.unique(np.random.choice(np.arange(len(l)), size=length))
    idx = np.sort(idx)
    return l[idx]


def numpy_to_tuple(function):
    def wrapper(*args):
        args = [tuple(list(x)) if type(x) == np.ndarray else x for x in args]
        result = function(*args)
        return result

    return wrapper
