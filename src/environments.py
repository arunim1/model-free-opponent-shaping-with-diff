import os.path as osp
import torch
from torch import nn
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_device(device) # type: ignore

def asymmetrize(p_m_1, p_m_2, eps=1e-3): 
    # [(2,2), (0,3+e), (3,0), (1,1+e)], i.e. changing player 2's incentive to defect
    p_m_1 = p_m_1.clone() # needed because of in-place operations + device apparently
    p_m_2 = p_m_2.clone()
    p_m_1 += torch.tensor([[0, eps], [0, eps]]).to(device)  # correct if p_m_1 corresponds to player 2's loss 
    p_m_2 += torch.tensor([[0, 0], [0, 0]]).to(device)
    return p_m_1, p_m_2


def diff_nn(th_0, th_1, upper_bound=1.0, iterated=False):
    # output = torch.sum(torch.abs(th_0 - th_1), dim=-1, keepdim=True)
    # output = torch.norm(th_0 - th_1, dim=-1, keepdim=True)
    """
    we start by simply saying that the diff value *will be* in the range [0, 1]
    this means each policy th_0, th_1 can be said to map a value in [0, 1] to a value in [0, 1], where the output is a p(cooperate)
    or for iterated games, mapping a value in [0, 1] to a 5-vector with each entry in [0, 1] 
    """

    bs = th_0.shape[0]

    upper_bound = torch.tensor(upper_bound).to(device)

    # Generate an array of inputs within the range [0, 1].
    diff_inputs = (torch.rand(100) * upper_bound).unsqueeze(1).unsqueeze(1).to(device)

    # Repeat diff_inputs to make it have shape: (bs, 100, 1, 1)
    diff_inputs_repeated = diff_inputs.repeat(bs, 1, 1, 1)

    def calculate_neurons(n_params):
        # Coefficients for the quadratic equation
        a = 2
        b = 9 if iterated else 5
        c = -(n_params - 5) if iterated else -(n_params - 1)
        
        # Calculate the discriminant
        discriminant = b**2 - 4*a*c
        
        # If discriminant is negative, then no real solution exists
        if discriminant < 0:
            return None
        
        # Calculate the two possible solutions using the quadratic formula
        neuron1 = (-b + math.sqrt(discriminant)) / (2*a)
        neuron2 = (-b - math.sqrt(discriminant)) / (2*a)
        
        # Return the positive solution as the number of neurons should be positive
        return math.floor(max(neuron1, neuron2))

    n_neurons = calculate_neurons(th[0].shape[1])

    W1_1, W1_2 = th_0[:, 0:n_neurons].unsqueeze(1).unsqueeze(-1), th_1[:, 0:n_neurons].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, n_neurons, 1)
    b1_1, b1_2 = th_0[:, n_neurons:2*n_neurons].unsqueeze(1).unsqueeze(-1), th_1[:, n_neurons:2*n_neurons].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, n_neurons, 1)
    W2_1, W2_2 = th_0[:, 2*n_neurons:2*n_neurons + n_neurons**2].reshape((bs, n_neurons, n_neurons)), th_1[:, 2*n_neurons:2*n_neurons + n_neurons**2].reshape((bs, n_neurons, n_neurons)) # has length n_neurons**2
    b2_1, b2_2 = th_0[:, 2*n_neurons + n_neurons**2:2*n_neurons + n_neurons**2 + n_neurons].unsqueeze(1), th_1[:, 2*n_neurons + n_neurons**2:2*n_neurons + n_neurons**2 + n_neurons].unsqueeze(1) # has length n_neurons
    W3_1, W3_2 = th_0[:, 2*n_neurons + n_neurons**2 + n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + n_neurons].reshape((bs, n_neurons, n_neurons)), th_1[:, 2*n_neurons + n_neurons**2 + n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + n_neurons].reshape((bs, n_neurons, n_neurons)) # has length n_neurons**2
    b3_1, b3_2 = th_0[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons].unsqueeze(1), th_1[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons].unsqueeze(1) # has length n_neurons
    if iterated: 
        W4_1, W4_2 = th_0[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + 5*n_neurons].reshape((bs, n_neurons, 5*n_neurons)), th_1[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + 5*n_neurons].reshape((bs, n_neurons, 5*n_neurons)) # has length 5*n_neurons**2
        b4_1, b4_2 = th_0[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + 5*n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + 5*n_neurons + 5].unsqueeze(1), th_1[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + 5*n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + 5*n_neurons + 5].unsqueeze(1) # has length 5
    else:
        W4_1, W4_2 = th_0[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + n_neurons].reshape((bs, n_neurons, n_neurons)), th_1[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + n_neurons].reshape((bs, n_neurons, n_neurons)) # has length n_neurons**2
        b4_1, b4_2 = th_0[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + n_neurons + n_neurons].unsqueeze(1), th_1[:, 2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + n_neurons:2*n_neurons + n_neurons**2 + 2*n_neurons**2 + 2*n_neurons + n_neurons + n_neurons].unsqueeze(1) # has length n_neurons
    
    x1_1, x1_2 = torch.relu(diff_inputs_repeated * W1_1 + b1_1), torch.relu(diff_inputs_repeated * W1_2 + b1_2)  # each has size (bs, 100, 40, 1)
    x1_1_u, x1_2_u = x1_1.squeeze(-1), x1_2.squeeze(-1)  # each has size (bs, 1, 100, 40)
    b2_1_u, b2_2_u = b2_1.unsqueeze(1), b2_2.unsqueeze(1)
    x2_1, x2_2 = torch.relu(torch.matmul(x1_1_u, W2_1) + b2_1_u), torch.relu(torch.matmul(x1_2_u, W2_2) + b2_2_u)
    x2_1_u, x2_2_u = x2_1.squeeze(1), x2_2.squeeze(1)  # each has size (bs, 100, 40)
    b3_1_u, b3_2_u = b3_1.unsqueeze(1), b3_2.unsqueeze(1)
    x3_1, x3_2 = torch.relu(torch.matmul(x2_1_u, W3_1) + b3_1_u), torch.relu(torch.matmul(x2_2_u, W3_2) + b3_2_u)
    x3_1_u, x3_2_u = x3_1.squeeze(1), x3_2.squeeze(1)  # each has size (bs, 100, 40)
    b4_1_u, b4_2_u = b4_1.unsqueeze(1), b4_2.unsqueeze(1)
    x4_1, x4_2 = torch.matmul(x3_1_u, W4_1) + b4_1_u, torch.matmul(x3_2_u, W4_2) + b4_2_u
    p_1, p_2 = torch.sigmoid(x4_1), torch.sigmoid(x4_2)

    if p_1.shape[2] == 5: # size is (bs, 100, 5)
        diff_output = torch.mean(torch.abs(p_1 - p_2), dim=(1,2)).unsqueeze(-1)
        # has size (bs, 1)
    else: # size is (bs, 100, 1)
        diff_output = torch.mean(torch.abs(p_1 - p_2), dim=1)

    return diff_output


def def_Ls_NN(p_m_1, p_m_2, bs, gamma_inner=0.96, iterated=False, diff_game=False): # works for ~arbitrary number of params
    def Ls(th): # th is a list of two tensors, each of shape (bs, params) for iterated games 
        th[0] = th[0].clone().to(device) # really not quite sure why this is necessary but it is. 
        th[1] = th[1].clone().to(device)
        nn_upper_bound = 0.2

        if diff_game: 
            
            def calculate_neurons(n_params):
                # Coefficients for the quadratic equation
                a = 2
                b = 9 if iterated else 5
                c = -(n_params - 5) if iterated else -(n_params - 1)
                
                # Calculate the discriminant
                discriminant = b**2 - 4*a*c
                
                # If discriminant is negative, then no real solution exists
                if discriminant < 0:
                    return None
                
                # Calculate the two possible solutions using the quadratic formula
                neuron1 = (-b + math.sqrt(discriminant)) / (2*a)
                neuron2 = (-b - math.sqrt(discriminant)) / (2*a)
                
                # Return the positive solution as the number of neurons should be positive, and round it down.
                return math.floor(max(neuron1, neuron2))

            n_neurons = calculate_neurons(th[0].shape[1])
            diff = diff_nn(th[0], th[1], upper_bound=nn_upper_bound, iterated=iterated) # has shape (bs, 1)
            
            diff1 = diff + 0.1 * torch.rand_like(diff)
            diff2 = diff + 0.1 * torch.rand_like(diff) 


            W1_1, W1_2 = th[0][:, 0:n_neurons], th[1][:, 0:n_neurons] # has length n_neurons
            b1_1, b1_2 = th[0][:, n_neurons:2*n_neurons], th[1][:, n_neurons:2*n_neurons]
            
            x1_1, x1_2 = torch.relu(diff1 * W1_1 + b1_1), torch.relu(diff2 * W1_2 + b1_2)
            # second layer
            W2_1, W2_2 = th[0][:, 2*n_neurons:2*n_neurons + n_neurons*n_neurons].reshape((bs, n_neurons, n_neurons)), th[1][:, 2*n_neurons:2*n_neurons + n_neurons*n_neurons].reshape((bs, n_neurons, n_neurons))
            b2_1, b2_2 = th[0][:, 2*n_neurons + n_neurons*n_neurons:2*n_neurons + n_neurons*n_neurons + n_neurons], th[1][:, 2*n_neurons + n_neurons*n_neurons:2*n_neurons + n_neurons*n_neurons + n_neurons] # has length n_neurons
            b2_1_u, b2_2_u = b2_1.unsqueeze(1), b2_2.unsqueeze(1)
            x1_1, x1_2 = x1_1.unsqueeze(1), x1_2.unsqueeze(1)
            x2_1, x2_2 = torch.relu(torch.bmm(x1_1, W2_1) + b2_1_u), torch.relu(torch.bmm(x1_2, W2_2) + b2_2_u)

            # third layer, also n_neurons x n_neurons
            W3_1, W3_2 = th[0][:, 2*n_neurons + n_neurons*n_neurons + n_neurons:2*n_neurons + 2*n_neurons*n_neurons + n_neurons].reshape((bs, n_neurons, n_neurons)), th[1][:, 2*n_neurons + n_neurons*n_neurons + n_neurons:2*n_neurons + 2*n_neurons*n_neurons + n_neurons].reshape((bs, n_neurons, n_neurons))
            b3_1, b3_2 = th[0][:, 2*n_neurons + 2*n_neurons*n_neurons + n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons], th[1][:, 2*n_neurons + 2*n_neurons*n_neurons + n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons] # has length n_neurons
            b3_1_u, b3_2_u = b3_1.unsqueeze(1), b3_2.unsqueeze(1)
            x3_1, x3_2 = torch.relu(torch.bmm(x2_1, W3_1) + b3_1_u), torch.relu(torch.bmm(x2_2, W3_2) + b3_2_u)
            
            if iterated: 
                # fourth layer, final, n_neurons x 5
                W4_1, W4_2 = th[0][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons*5].reshape((bs, n_neurons, 5)), th[1][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons*5].reshape((bs, n_neurons, 5))
                b4_1, b4_2 = th[0][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons*5:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons*5 + 5], th[1][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons*5:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons*5 + 5] # has length 5
                b4_1_u, b4_2_u = b4_1.unsqueeze(1), b4_2.unsqueeze(1)
                x4_1, x4_2 = torch.bmm(x3_1, W4_1) + b4_1_u, torch.bmm(x3_2, W4_2) + b4_2_u
                x4_1, x4_2 = x4_1.squeeze(1), x4_2.squeeze(1) # outputs of NN

                p_1_0, p_2_0 = torch.sigmoid(x4_1[:, 0:1]), torch.sigmoid(x4_2[:, 0:1])
                p_1 = torch.reshape(torch.sigmoid(x4_1[:, 1:5]), (bs, 4, 1))
                p_2 = torch.reshape(torch.sigmoid(x4_1[:, 1:5]), (bs, 4, 1))
            else:
                # fourth layer, final, n_neurons x 1
                W4_1 = th[0][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons].reshape((bs, n_neurons, 1))
                W4_2 = th[1][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons].reshape((bs, n_neurons, 1))
                b4_1 = th[0][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons + 1]
                b4_2 = th[1][:, 2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons:2*n_neurons + 2*n_neurons*n_neurons + 2*n_neurons + n_neurons + 1]
                b4_1_u, b4_2_u = b4_1.unsqueeze(1), b4_2.unsqueeze(1)
                x4_1, x4_2 = torch.bmm(x3_1, W4_1) + b4_1_u, torch.bmm(x3_2, W4_2) + b4_2_u
                x4_1, x4_2 = x4_1.squeeze(1), x4_2.squeeze(1) # outputs of NN

                p_1, p_2 = torch.sigmoid(x4_1), torch.sigmoid(x4_2)
        else:
            if iterated: 
                p_1_0, p_2_0 = torch.sigmoid(th[0][:, 0:1]), torch.sigmoid(th[1][:, 0:1])
                p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
                p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
            else:
                p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])

        if iterated:
            p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
            P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)
            M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
            L_1 = -torch.matmul(M, torch.reshape(p_m_1, (bs, 4, 1))) # player 2's loss, since p2's params are th[0]
            L_2 = -torch.matmul(M, torch.reshape(p_m_2, (bs, 4, 1))) # player 1's loss, since p1's params are th[1]
            return [L_1.squeeze(-1), L_2.squeeze(-1), M]
        else: 
            x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
            M = torch.bmm(x.view(bs, 2, 1), y.view(bs, 1, 2)).view(bs, 1, 4)
            L_1 = -torch.matmul(M, torch.reshape(p_m_1, (bs, 4, 1))) # same outputs as old version, but with M
            L_2 = -torch.matmul(M, torch.reshape(p_m_2, (bs, 4, 1)))
            return [L_1.squeeze(-1), L_2.squeeze(-1), M]
    
    return Ls


def def_Ls_threshold_game(p_m_1, p_m_2, bs, gamma_inner=0.96, iterated=False, diff_game=False):
    def Ls(th): # th is a list of two tensors, each of shape (bs, 5) for iterated games and (bs, 1) for one-shot games
        # sig(th) = probabilities or thresholds, depending. 
        if iterated:
            t_1_0 = torch.sigmoid(th[0][:, 0:1])
            t_2_0 = torch.sigmoid(th[1][:, 0:1])
            t_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
            t_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
            if diff_game: 
                diff_0 = torch.abs(t_1_0 - t_2_0)
                p_1_0 = torch.relu(t_1_0 - diff_0)
                p_2_0 = torch.relu(t_2_0 - diff_0)
                diff = torch.abs(t_1 - t_2)
                p_1 = torch.relu(t_1 - diff)
                p_2 = torch.relu(t_2 - diff)
            else:
                p_1_0, p_2_0, p_1, p_2 = t_1_0, t_2_0, t_1, t_2

            p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
            P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)
            M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))  
            L_1 = -torch.matmul(M, torch.reshape(p_m_1, (bs, 4, 1))) # player 2's loss, since p2's params are th[0]
            L_2 = -torch.matmul(M, torch.reshape(p_m_2, (bs, 4, 1))) # player 1's loss, since p1's params are th[1]
            return [L_1.squeeze(-1), L_2.squeeze(-1), M] 

        else: 
            if diff_game:
                t_1, t_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
                diff = torch.abs(t_1 - t_2)
                p_1 = torch.relu(t_1 - diff)
                p_2 = torch.relu(t_2 - diff)
            else:
                p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
            
            x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
            M = torch.bmm(x.view(bs, 2, 1), y.view(bs, 1, 2)).view(bs, 1, 4)
            L_1 = -torch.matmul(M, torch.reshape(p_m_1, (bs, 4, 1))) # same as old version, but with M
            L_2 = -torch.matmul(M, torch.reshape(p_m_2, (bs, 4, 1))) 
            # L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), p_m_1), y.unsqueeze(-1))
            # L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), p_m_2), y.unsqueeze(-1))
            return [L_1.squeeze(-1), L_2.squeeze(-1), M]
    
    return Ls


def sl_batched(bs, gamma_inner=0.96, iterated=False, diff_game=False, asym=None):
    # silly game: you get -1 if you defect and 1 if you cooperate. regardless of what the other player does.
    dims = [5, 5] if iterated else [1, 1]

    payout_mat_1 = torch.Tensor([[1, 1], [-1, -1]]).to(device)

    payout_mat_2 = payout_mat_1.T
    if asym is not None:
        payout_mat_1, payout_mat_2 = asymmetrize(payout_mat_1, payout_mat_2, asym)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1)

    Ls = def_Ls(p_m_1=payout_mat_1, p_m_2=payout_mat_2, bs=bs, gamma_inner=gamma_inner, iterated=iterated, diff_game=diff_game)

    return dims, Ls

def pd_batched(bs, gamma_inner=0.96, iterated=False, diff_game=False, asym=None):
    dims = [5, 5] if iterated else [1, 1]
    G = 2.5
    payout_mat_1 = torch.Tensor([[G, 0], [G + 1, 1]]).to(device)
    payout_mat_1 -= 3
    # payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]]).to(device)

    payout_mat_2 = payout_mat_1.T
    if asym is not None:
        payout_mat_1, payout_mat_2 = asymmetrize(payout_mat_1, payout_mat_2, asym)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1)

    Ls = def_Ls(p_m_1=payout_mat_1, p_m_2=payout_mat_2, bs=bs, gamma_inner=gamma_inner, iterated=iterated, diff_game=diff_game)

    return dims, Ls

def mp_batched(bs, gamma_inner=0.96, iterated=False, diff_game=False, asym=None):
    dims = [5, 5] if iterated else [1, 1]
    payout_mat_1 = torch.Tensor([[-1, 1], [1, -1]]).to(device)
    payout_mat_2 = -payout_mat_1
    if asym is not None: # unsure about MP asymmetry
        payout_mat_1, payout_mat_2 = asymmetrize(payout_mat_1, payout_mat_2, asym)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1)

    Ls = def_Ls(p_m_1=payout_mat_1, p_m_2=payout_mat_2, bs=bs, gamma_inner=gamma_inner, iterated=iterated, diff_game=diff_game)

    return dims, Ls

def hd_batched(bs, gamma_inner=0.96, iterated=False, diff_game=False, asym=None):
    dims = [5, 5] if iterated else [1, 1]
    payout_mat_1 = torch.Tensor([[0, -1], [1, -100]]).to(device)
    payout_mat_2 = payout_mat_1.T
    if asym is not None:
        payout_mat_1, payout_mat_2 = asymmetrize(payout_mat_1, payout_mat_2, asym)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1)

    Ls = def_Ls(p_m_1=payout_mat_1, p_m_2=payout_mat_2, bs=bs, gamma_inner=gamma_inner, iterated=iterated, diff_game=diff_game)

    return dims, Ls

def sh_batched(bs, gamma_inner=0.96, iterated=False, diff_game=False, asym=None):
    dims = [5, 5] if iterated else [1, 1]
    payout_mat_1 = torch.Tensor([[3, 0], [1, 1]]).to(device)
    payout_mat_2 = payout_mat_1.T
    if asym is not None:
        payout_mat_1, payout_mat_2 = asymmetrize(payout_mat_1, payout_mat_2, asym)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1)
    
    Ls = def_Ls(p_m_1=payout_mat_1, p_m_2=payout_mat_2, bs=bs, gamma_inner=gamma_inner, iterated=iterated, diff_game=diff_game)

    return dims, Ls


def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True, allow_unused=True)[0]
    return grad

def compute_best_response(outer_th_ba):
    batch_size = 1
    std = 0
    num_steps = 1000
    lr = 1

    ipd_batched_env = pd_batched(batch_size, gamma_inner=0.96, iterated=True)[1]
    inner_th_ba = torch.nn.init.normal_(torch.empty((batch_size, 5), requires_grad=True), std=std).to(device)
    for i in range(num_steps):
        th_ba = [inner_th_ba, outer_th_ba.detach()]
        l1, l2, M = ipd_batched_env(th_ba)
        grad = get_gradient(l1.sum(), inner_th_ba)
        with torch.no_grad():
            inner_th_ba -= grad * lr
    print(l1.mean() * (1 - 0.96))
    return inner_th_ba

def generate_mamaml(b, d, inner_env, game, inner_lr=1):
    """
    This is an improved version of the algorithm presented in this paper:
    https://arxiv.org/pdf/2011.00382.pdf
    Rather than calculating the loss using multiple policy gradients terms,
    this approach instead directly takes all of the gradients through because the environment is differentiable.
    """
    outer_lr = 0.01
    mamaml = torch.nn.init.normal_(torch.empty((1, d), requires_grad=True, device=device), std=1.0)
    alpha = torch.rand(1, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([mamaml, alpha], lr=outer_lr)

    for ep in range(1000):
        agent = mamaml.clone().repeat(b, 1)
        opp = torch.nn.init.normal_(torch.empty((b, d), requires_grad=True), std=1.0).to(device)
        total_agent_loss = 0
        total_opp_loss = 0
        for step in range(100):
            l1, l2, M = inner_env([opp, agent])
            total_agent_loss = total_agent_loss + l2.sum()
            total_opp_loss = total_opp_loss + l1.sum()

            opp_grad = get_gradient(l1.sum(), opp)
            agent_grad = get_gradient(l2.sum(), agent)
            opp = opp - opp_grad * inner_lr
            agent = agent - agent_grad * alpha

        optimizer.zero_grad()
        total_agent_loss.sum().backward()
        optimizer.step()
        print(total_agent_loss.sum().item())

    torch.save((mamaml, alpha), f"mamaml_{game}.th")


class MetaGames:
    def __init__(self, b, opponent="NL", game="IPD", mmapg_id=0, nn_game=False):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn.
        COPYCAT = Copies what opponent played last step.
        """
        self.gamma_inner = 0.96
        self.b = b

        self.game = game

        self.diff_game = True if game.find("diff") != -1 or game.find("Diff") != -1 else False
        self.iterated = True if game.find("I") != -1 else False
        # self.ccdr = ccdr
        self.nn_game = nn_game
        # self.adam = True

        global def_Ls 
        def_Ls = def_Ls_NN if nn_game else def_Ls_threshold_game

        if game.find("PD") != -1:
            d, self.game_batched = pd_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 1
        elif game.find("MP") != -1:
            d, self.game_batched = mp_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 0.1
        elif game.find("HD") != -1:
            d, self.game_batched = hd_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 1
        elif game.find("SH") != -1:
            d, self.game_batched = sh_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 1
        else:
            raise NotImplementedError
        
        self.std = 1
        self.d = d[0]

        if self.nn_game and self.diff_game: self.d = 3565 if self.iterated else 3401 # 40, 7
        if self.nn_game and self.diff_game: self.d = 40 if self.iterated else 7

        self.opponent = opponent
        if self.opponent == "MAMAML":
            f = f"data/mamaml_{self.game}_{mmapg_id}.th"
            assert osp.exists(f), "Generate the MAMAML weights first"
            self.init_th_ba = torch.load(f)
        else:
            self.init_th_ba = None

    def reset(self, info=False):
        if self.init_th_ba is not None:
            self.inner_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)
        else:
            if self.nn_game and self.diff_game: 
                self.inner_th_ba = torch.nn.init.xavier_normal_(torch.empty((self.b, self.d), requires_grad=True)).to(device)
            else:
                self.inner_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        if self.nn_game and self.diff_game: 
            outer_th_ba = torch.nn.init.xavier_normal_(torch.empty((self.b, self.d), requires_grad=True)).to(device)
        else:
            outer_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        state, _, _, M = self.step(outer_th_ba)
        if info:
            return state, M
        else:
            return state

    def step(self, outer_th_ba):
        last_inner_th_ba = self.inner_th_ba.detach().clone()
        if self.opponent == "NL" or self.opponent == "MAMAML":
            th_ba = [self.inner_th_ba, outer_th_ba.detach()]
            l1, l2, M = self.game_batched(th_ba)
            grad = get_gradient(l1.sum(), self.inner_th_ba)
            with torch.no_grad():
                self.inner_th_ba -= grad * self.lr
        elif self.opponent == "LOLA":
            th_ba = [self.inner_th_ba, outer_th_ba.detach()]
            th_ba[1].requires_grad = True
            l1, l2, M = self.game_batched(th_ba)
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)]
            term = (grad_L[1][0] * grad_L[1][1]).sum()
            grad = grad_L[0][0] - self.lr * get_gradient(term, th_ba[0])
            with torch.no_grad():
                self.inner_th_ba -= grad * self.lr
        elif self.opponent == "BR":
            num_steps = 1000
            inner_th_ba = torch.nn.init.normal_(torch.empty((self.b, 5), requires_grad=True), std=self.std).to(device)
            for i in range(num_steps):
                th_ba = [inner_th_ba, outer_th_ba.detach()]
                l1, l2, M = self.game_batched(th_ba)
                grad = get_gradient(l1.sum(), inner_th_ba)
                with torch.no_grad():
                    inner_th_ba -= grad * self.lr
            with torch.no_grad():
                self.inner_th_ba = inner_th_ba
                th_ba = [self.inner_th_ba, outer_th_ba.detach()]
                l1, l2, M = self.game_batched(th_ba)
        else:
            raise NotImplementedError

        if self.iterated:
            return torch.sigmoid(torch.cat((outer_th_ba, last_inner_th_ba), dim=-1)).detach(), (-l2 * (1 - self.gamma_inner)).detach(), (-l1 * (1 - self.gamma_inner)).detach(), M
        else:
            return torch.sigmoid(torch.cat((outer_th_ba, last_inner_th_ba), dim=-1)).detach(), -l2.detach(), -l1.detach(), M


class SymmetricMetaGames:
    def __init__(self, b, game="IPD", nn_game=False):
        self.gamma_inner = 0.96

        self.b = b
        self.game = game

        self.diff_game = True if game.find("diff") != -1 or game.find("Diff") != -1 else False
        self.iterated = True if game.find("I") != -1 else False

        global def_Ls 
        def_Ls = def_Ls_NN if nn_game else def_Ls_threshold_game

        if game.find("PD") != -1:
            d, self.game_batched = pd_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 1
        elif game.find("MP") != -1:
            d, self.game_batched = mp_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 0.1
        elif game.find("HD") != -1:
            d, self.game_batched = hd_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 1
        elif game.find("SH") != -1:
            d, self.game_batched = sh_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game)
            self.lr = 1
        else:
            raise NotImplementedError
        
        self.std = 1
        self.d = d[0]

        if nn_game and self.diff_game: self.d = 3565 if self.iterated else 3401 # 40, 7


    def reset(self, info=False):
        p_ba_0 = torch.nn.init.normal_(torch.empty((self.b, self.d)), std=self.std).to(device)
        p_ba_1 = torch.nn.init.normal_(torch.empty((self.b, self.d)), std=self.std).to(device)
        state_0 = torch.sigmoid(torch.cat((p_ba_0.detach(), p_ba_1.detach()), dim=-1))
        state_1 = torch.sigmoid(torch.cat((p_ba_1.detach(), p_ba_0.detach()), dim=-1))

        if info:
            state, _, M = self.step(p_ba_0, p_ba_1)
            return state, M
        else:
            return [state_0, state_1]

    def step(self, p_ba_0, p_ba_1):
        th_ba = [p_ba_0.detach(), p_ba_1.detach()]
        l1, l2, M = self.game_batched(th_ba)
        state_0 = torch.sigmoid(torch.cat((p_ba_0.detach(), p_ba_1.detach()), dim=-1))
        state_1 = torch.sigmoid(torch.cat((p_ba_1.detach(), p_ba_0.detach()), dim=-1))

        if self.iterated:
            return [state_0, state_1], [-l1 * (1 - self.gamma_inner), -l2 * (1 - self.gamma_inner)], M
        else:
            return [state_0, state_1], [-l1.detach(), -l2.detach()], M


class NonMfosMetaGames:
    def __init__(self, b, p1="NL", p2="NL", game="IPD", lr=None, mmapg_id=None, asym=None, nn_game=False, ccdr=False, adam=False):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn. Used for sanity checking.
        """
        self.gamma_inner = 0.96
        self.b = b

        self.p1 = p1
        self.p2 = p2
        self.game = game
        self.ccdr = ccdr
        self.nn_game = nn_game
        self.adam = adam
        
        global def_Ls 
        def_Ls = def_Ls_NN if self.nn_game else def_Ls_threshold_game

        self.diff_game = True if game.find("diff") != -1 or game.find("Diff") != -1 else False
        self.iterated = True if game.find("I") != -1 else False

        if game.find("PD") != -1:
            d, self.game_batched = pd_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game, asym=asym)
            self.lr = 1
        elif game.find("MP") != -1:
            d, self.game_batched = mp_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game, asym=asym)
            self.lr = 0.1
        elif game.find("HD") != -1:
            d, self.game_batched = hd_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game, asym=asym)
            # self.lr = 0.1 if self.diff_game else 0.01 
            self.lr = 1
        elif game.find("SH") != -1:
            d, self.game_batched = sh_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game, asym=asym)
            # self.lr = 3.612
            # self.lr *= 0.01 if self.diff_game and self.iterated else 0.1 if self.diff_game else 1
            self.lr = 1
        elif game.find("SL") != -1:
            d, self.game_batched = sl_batched(b, gamma_inner=self.gamma_inner, iterated=self.iterated, diff_game=self.diff_game, asym=asym)
            self.lr = 1
        else:
            raise NotImplementedError

        self.std = 1
        if lr is not None:
            # first testing if lr is a number or a list
            if isinstance(lr, list):
                if len(lr) == 2:
                    self.lr = lr[0]
                    self.lr_2 = lr[1]
            else:
                self.lr = lr
                self.lr_2 = lr

        self.d = d[0]

        if self.adam:
            self.m_1 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.v_1 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.beta1_1 = 0.9
            self.beta2_1 = 0.999
            self.eps_1 = 1e-8
            self.t_1 = 0
            self.m_2 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.v_2 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.beta1_2 = 0.9
            self.beta2_2 = 0.999
            self.eps_2 = 1e-8
            self.t_2 = 0
        
        n_neurons = 8 
        n_params_5 = 2*(n_neurons**2) + 9*n_neurons + 5
        n_params_1 = 2*(n_neurons**2) + 5*n_neurons + 1
        if self.nn_game and self.diff_game: self.d = n_params_5 if self.iterated else n_params_1 # (3565, 3401), (40, 7)

        self.init_th_ba = None
        if self.p1 == "MAMAML" or self.p2 == "MAMAML":
            if self.init_th_ba is not None:
                raise NotImplementedError
            f = f"data/mamaml_{self.game}_{mmapg_id}.th"
            assert osp.exists(f), "Generate the MAMAML weights first"
            # print(f"GENERATING MAPG WEIGHTS TO {f}")
            # generate_meta_mapg(self.b, self.d, self.game_batched, self.game, inner_lr=self.lr)
            self.init_th_ba = torch.load(f)
            print(self.init_th_ba)

    def reset(self, info=False):
        # self.p1_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        # self.p2_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        # using xavier_normal_ for nn games
        if self.nn_game and self.diff_game: 
            self.p1_th_ba = torch.nn.init.xavier_normal_(torch.empty((self.b, self.d), requires_grad=True)).to(device)
            self.p2_th_ba = torch.nn.init.xavier_normal_(torch.empty((self.b, self.d), requires_grad=True)).to(device)
        else:
            self.p1_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
            self.p2_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)

        if self.p1 == "MAMAML":
            self.p1_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)
        if self.p2 == "MAMAML":
            self.p2_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)
        
        if self.adam:
            self.m_1 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.v_1 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.beta1_1 = 0.9
            self.beta2_1 = 0.999
            self.eps_1 = 1e-8
            self.t_1 = 0
            self.m_2 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.v_2 = torch.zeros((self.b, self.d), requires_grad=False).to(device)
            self.beta1_2 = 0.9
            self.beta2_2 = 0.999
            self.eps_2 = 1e-8
            self.t_2 = 0
        
        state, _, _, M = self.step()
        if info:
            return state, M

        return None
    
    def update(self, p1or2, grad):
        if p1or2 == 1:
            with torch.no_grad():
                if self.adam:
                    self.t_1 += 1
                    self.m_1 = self.beta1_1 * self.m_1 + (1 - self.beta1_1) * grad
                    self.v_1 = self.beta2_1 * self.v_1 + (1 - self.beta2_1) * (grad ** 2)
                    m_hat = self.m_1 / (1 - self.beta1_1 ** self.t_1) # bias correction
                    v_hat = self.v_1 / (1 - self.beta2_1 ** self.t_1)
                    self.p1_th_ba -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps_1) 
                else: 
                    self.p1_th_ba -= grad * self.lr
        elif p1or2 == 2:
            with torch.no_grad():
                if self.adam:
                    self.t_2 += 1
                    self.m_2 = self.beta1_2 * self.m_2 + (1 - self.beta1_2) * grad
                    self.v_2 = self.beta2_2 * self.v_2 + (1 - self.beta2_2) * (grad ** 2)
                    m_hat = self.m_2 / (1 - self.beta1_2 ** self.t_2)
                    v_hat = self.v_2 / (1 - self.beta2_2 ** self.t_2)
                    self.p2_th_ba -= m_hat / (torch.sqrt(v_hat) + self.eps_2) * self.lr
                else:
                    self.p2_th_ba -= grad * self.lr
        else:
            raise NotImplementedError

    def step(self, info=False):
        last_p1_th_ba = self.p1_th_ba.clone()
        last_p2_th_ba = self.p2_th_ba.clone()
        th_ba = [self.p2_th_ba, self.p1_th_ba]
        
        l1_reg, l2_reg, M = self.game_batched(th_ba) # l2 is for p1 aka p1_th_ba, l1 is for p2 aka p2_th_ba
        
        if self.ccdr: 
            th_CC1 = [self.p1_th_ba, self.p1_th_ba]
            th_CC2 = [self.p2_th_ba, self.p2_th_ba]
            if self.nn_game and self.diff_game:
                th_DR1 = [self.p2_th_ba, torch.nn.init.xavier_normal_(torch.empty_like(self.p1_th_ba))]  
                th_DR2 = [torch.nn.init.xavier_normal_(torch.empty_like(self.p2_th_ba)), self.p1_th_ba]
            else:
                th_DR1 = [self.p2_th_ba, torch.nn.init.normal_(torch.empty_like(self.p1_th_ba), std=self.std)]  
                th_DR2 = [torch.nn.init.normal_(torch.empty_like(self.p2_th_ba), std=self.std), self.p1_th_ba]
            _, l2_CC1, _ = self.game_batched(th_CC1)
            l1_CC2, _, _ = self.game_batched(th_CC2)
            l1_DR1, _, _ = self.game_batched(th_DR1)
            _, l2_DR2, _ = self.game_batched(th_DR2)
            l1 = (l1_reg + l1_CC2 + l1_DR1)/3  
            l2 = (l2_reg + l2_CC1 + l2_DR2)/3  
        else:
            l1, l2 = l1_reg, l2_reg

        # UPDATE P1
        if self.p1 == "NL" or self.p1 == "MAMAML":
            grad = get_gradient(l2.sum(), self.p1_th_ba)
            self.update(1, grad)
        elif self.p1 == "LOLA":
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)]
            term = (grad_L[0][0] * grad_L[0][1]).sum()
            grad = grad_L[1][1] - self.lr_2 * get_gradient(term, th_ba[1])
            self.update(1, grad)
        elif self.p1 == "STATIC":
            pass
        else:
            raise NotImplementedError
        
        # UPDATE P2
        if self.p2 == "NL" or self.p2 == "MAMAML":
            grad = get_gradient(l1.sum(), self.p2_th_ba)
            self.update(2, grad)
        elif self.p2 == "LOLA":
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)] 
            term = (grad_L[1][0] * grad_L[1][1]).sum()
            grad = grad_L[0][0] - self.lr_2 * get_gradient(term, th_ba[0])
            self.update(2, grad)
        elif self.p2 == "STATIC":
            pass
        else:
            raise NotImplementedError
        
        if self.iterated:
            return torch.sigmoid(torch.cat([last_p1_th_ba, last_p2_th_ba])).detach(), (-l2 * (1 - self.gamma_inner)).detach(), (-l1 * (1 - self.gamma_inner)).detach(), M
        else:
            return torch.sigmoid(torch.cat([last_p1_th_ba, last_p2_th_ba])).detach(), -l2.detach(), -l1.detach(), M
        
    def fwd_step(self, p1_th=None, p2_th=None): 
        # just gets the rewards and plays the game, doesn't update (clone and detach evertyhing just in case)
        if p1_th is None: p1_th = self.p1_th_ba.detach().clone()
        if p2_th is None: p2_th = self.p2_th_ba.detach().clone()
        th_ba = [p1_th, p2_th]
        l1, l2, M = self.game_batched(th_ba) # here, l2 corresponds to p1_th_ba, l1 corresponds to p2_th_ba

        return torch.sigmoid(torch.cat([self.p1_th_ba.detach(), self.p2_th_ba.detach()])).detach(), -l2.detach(), -l1.detach(), M

