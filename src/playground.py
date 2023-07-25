import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch import nn

def main(filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assume example shapes for th_0 and th_1 for demonstration purposes
    th_0 = torch.randn((5, 10))
    th_1 = torch.randn((5, 10))

    bs = th_0.shape[0]

    # Generate an array of inputs within the range [0, 1].
    diff_inputs = torch.linspace(0, 1, 100).unsqueeze(1).unsqueeze(1)

    # Repeat diff_inputs to make it have shape: (bs, 100, 1, 1)
    diff_inputs_repeated = diff_inputs.repeat(bs, 1, 1, 1)

    # First layer
    W1_1, W1_2 = th_0[:, 0:2].unsqueeze(1).unsqueeze(-1), th_1[:, 0:2].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, 2, 1)
    b1_1, b1_2 = th_0[:, 2:4].unsqueeze(1).unsqueeze(-1), th_1[:, 2:4].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, 2, 1)

    # Second layer
    W2_1, W2_2 = th_0[:, 4:6].reshape((bs, 2, 1)), th_1[:, 4:6].reshape((bs, 2, 1))
    b2_1, b2_2 = th_0[:, 6:7], th_1[:, 6:7]  # each has size (bs, 1)
    x1_1, x1_2 = torch.relu(diff_inputs_repeated * W1_1 + b1_1), torch.relu(diff_inputs_repeated * W1_2 + b1_2)  # each has size (bs, 100, 2, 1)

    b2_1_u, b2_2_u = b2_1.unsqueeze(1), b2_2.unsqueeze(1) 
    x1_1_u, x1_2_u = x1_1.squeeze(-1), x1_2.squeeze(-1)  # each has size (bs, 1, 100, 2)

    # Note: Using torch.matmul here instead of torch.bmm
    x2_1, x2_2 = torch.matmul(x1_1_u, W2_1) + b2_1_u, torch.matmul(x1_2_u, W2_2) + b2_2_u

    p_1, p_2 = torch.sigmoid(x2_1), torch.sigmoid(x2_2)
    print("functional:", x2_1.shape, x2_2.shape)
    print("detailed: ", x1_1_u.shape, W2_1.shape, b2_1_u.shape, x1_2_u.shape, W2_2.shape, b2_2_u.shape)

    print(p_1.shape, p_2.shape)

    out = torch.mean(torch.abs(p_1 - p_2), dim=1)
    print(out.shape)

    ## old version: 

    W1_1, W1_2 = th_0[:, 0:2], th_1[:, 0:2] # has size (bs, 2)
    b1_1, b1_2 = th_0[:, 2:4], th_1[:, 2:4] # has size (bs, 2)
    
    # second layer
    W2_1, W2_2 = th_0[:, 4:6].reshape((bs, 2, 1)), th_1[:, 4:6].reshape((bs, 2, 1))
    b2_1, b2_2 = th_0[:, 6:7], th_1[:, 6:7] # each has size (bs, 1)
    x1_1, x1_2 = torch.relu(diff_inputs[2] * W1_1 + b1_1), torch.relu(diff_inputs[2] * W1_2 + b1_2) # each has size (bs, 2)
    b2_1_u, b2_2_u = b2_1.unsqueeze(1), b2_2.unsqueeze(1) # each has size (bs, 1, 1
    x1_1_u, x1_2_u = x1_1.unsqueeze(1), x1_2.unsqueeze(1) # each has size (bs, 1, 2)
    x2_1, x2_2 = torch.bmm(x1_1_u, W2_1) + b2_1_u, torch.bmm(x1_2_u, W2_2) + b2_2_u
    x2_1_u, x2_2_u = x2_1.squeeze(1), x2_2.squeeze(1) # outputs of NN

    p_1, p_2 = torch.sigmoid(x2_1_u), torch.sigmoid(x2_2_u)
    print("old:", x2_1.shape, x2_2.shape)
    print("old:", x1_1_u.shape, W2_1.shape, b2_1_u.shape, x1_2_u.shape, W2_2.shape, b2_2_u.shape)
    print(p_1.shape, p_2.shape)

    

if __name__ == "__main__":
    main("mfos_ppo_IPD_NL_dumptest")
    