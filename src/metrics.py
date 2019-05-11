#!/usr/bin/env python

import numpy as np
import math

def get_hit_ratio_(ranked_list, target):
    return int(target in ranked_list)
def get_NDCG_(ranked_list, target):
    where = np.where(ranked_list == target)[0]
    return 0 if len(where) == 0 else math.log(2) / math.log(where[0] + 2)
def get_MRR_(ranked_list,target):
    try:
        return 1./(ranked_list.index(target) + 1)
    except:
        return 0

def get_hit_ratio(ranked_lists, targets):
    """
    Parameters
    ----------
    n: number of test instances
    k: number of items recommended
    ranked_lists: np.array of shape (n, k)
        ranked_lists[i,j] is the top j-th item recommended to i-th person
    targets: np.array of shape (n,)
        targets[i] is a i-th person's known positive
    Returns
    -------
    float
        hit-ratio score
    """
    return np.mean([get_hit_ratio_(ranked_lists[i], targets[i]) for i in range(len(targets))])

def get_NDCG(ranked_lists, targets):
    """
    Parameters
    ----------
    n: number of test instances
    k: number of items recommended
    ranked_lists: np.array of shape (n, k)
        ranked_lists[i,j] is the top j-th item recommended to i-th person
    targets: np.array of shape (n,)
        targets[i] is a i-th person's known positive
    Returns
    -------
    float
        NDCG score
    """
    return np.mean([get_NDCG_(ranked_lists[i], targets[i]) for i in range(len(targets))])

def get_MRR(ranked_lists, targets):
    """
    Parameters
    ----------
    n: number of test instances
    k: number of items recommended
    ranked_lists: np.array of shape (n, k)
        ranked_lists[i,j] is the top j-th item recommended to i-th person
    targets: np.array of shape (n,)
        targets[i] is a i-th person's known positive
    Returns
    -------
    float
        MRR score
    """
    return np.mean([get_MRR_(ranked_lists[i], targets[i]) for i in range(len(targets))])
