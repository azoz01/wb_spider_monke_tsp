import numpy as np


def permutation_difference(perm_1: np.ndarray, perm_2: np.ndarray) -> callable:
    arg_1 = np.argsort(perm_1)
    return arg_1[perm_2]


def decompose_permutation_to_swap_sequence(permutation):
    swaps = []
    permutation = permutation.tolist()
    for i in range(len(permutation)):
        idx = permutation.index(i)
        swaps.append((i, idx))
        permutation[i], permutation[idx] = i, permutation[i]
    return np.array(swaps[-1::-1])


def swap_sequence_to_permutation(ss):
    result = np.arange(np.max(ss) + 1)
    for pos1, pos2 in ss:
        result[pos1], result[pos2] = result[pos2], result[pos1]
    return result


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
