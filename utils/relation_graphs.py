import numpy as np
import torch

def _calc_prob(y):
    marginal_1 = np.sum(y, 0)/len(y)
    marginal_0 = 1-marginal_1
    joint_11 = np.matmul(y.transpose(), y)/len(y)
    joint_01 = np.matmul(1-y.transpose(), y)/len(y)
    joint_10 = np.matmul(y.transpose(), 1-y)/len(y)
    joint_00 = np.matmul(1-y.transpose(), 1-y)/len(y)
    conditional_11 = joint_11/marginal_1  # probability of dim=0 (row) given dim=1 (column)
    conditional_01 = joint_01/marginal_1
    conditional_10 = joint_10/marginal_0
    conditional_00 = joint_00/marginal_0
    return {'marginal_1': marginal_1, 'marginal_0': marginal_0,
            'joint_11': joint_11, 'conditional_11': conditional_11,
            'joint_01': joint_01, 'conditional_01': conditional_01,
            'joint_10': joint_10, 'conditional_10': conditional_10,
            'joint_00': joint_00, 'conditional_00': conditional_00
            }

def _by_chi2_contingency(tgt_binary, alpha, return_skeleton=False):
    from scipy.stats import chi2_contingency

    y = np.array(tgt_binary)
    # cell counts
    count_11 = np.matmul(y.transpose(), y)
    count_10 = np.matmul(y.transpose(), 1 - y)
    count_01 = np.matmul(1 - y.transpose(), y)
    count_00 = np.matmul(1 - y.transpose(), 1 - y)

    # chi2 testing on pairwise dependencies
    num_vars = y.shape[1]
    p_value = np.ones([num_vars,num_vars])
    for i in range(num_vars):
        for j in range(num_vars):
            c = np.array([[count_11[i, j], count_10[i, j]], [count_01[i, j], count_00[i, j]]])
            if (np.sum(c, 0) > 0).all() and (np.sum(c, 1) > 0).all():
                _, p_value[i, j], _, _ = chi2_contingency(c)
            else:
                p_value[i,j] = 1

    skeleton = 1*(p_value <= alpha)

    if return_skeleton:
        np.fill_diagonal(skeleton, 1)
        return torch.tensor(skeleton).unsqueeze(0)
    else:
        # adjacency matrices for two relations
        p = _calc_prob(y)
        marginal = p['marginal_1']
        conditional = p['conditional_11']
        pulling = skeleton * (conditional > marginal[:, None])
        pushing = skeleton * (conditional < marginal[:, None])
        np.fill_diagonal(pulling, 0)
        np.fill_diagonal(pushing, 0)
        pulling = torch.tensor(pulling).unsqueeze(0).cuda()
        pushing = torch.tensor(pushing).unsqueeze(0).cuda()

        adjs = [pulling, pushing]

        return adjs
