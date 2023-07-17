import numpy as np
import matplotlib.pyplot as plt
import json
import torch

def main(filename):
    # Define some random tensors for testing
    bs = 10
    p_1 = torch.rand(bs,1)
    p_2 = torch.rand(bs,1)
    p_m_1 = torch.rand(2, 2).reshape((1, 2, 2)).repeat(bs, 1, 1)

    # Original code
    x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
    L_1_old = -torch.matmul(torch.matmul(x.unsqueeze(1), p_m_1), y.unsqueeze(-1))

    # Modified code
    x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
    print(x.shape, y.shape, p_m_1.shape)

    M = torch.bmm(x.view(bs, 2, 1), y.view(bs, 1, 2)).view(bs, 1, 4)
    print(M.shape)

    L_1_new = -torch.matmul(M, torch.reshape(p_m_1, (bs, 4, 1)))

    # Check if the results are the same
    print(torch.allclose(L_1_old, L_1_new))




if __name__ == "__main__":
    main("mfos_ppo_IPD_NL_dumptest")
    