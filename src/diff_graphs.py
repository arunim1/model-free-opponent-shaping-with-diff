import matplotlib.pyplot as plt
import torch 
import math
import os
import numpy as np

device = torch.device("cpu" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore

def nn_main(th_0, th_1, upper_bound=1.0, iterated=False, p1=None, p2=None, name="test"):
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
    diff_inputs = (torch.rand(100).to(device) * upper_bound).unsqueeze(1).unsqueeze(1).to(device)
    
    # inv sigmoid of diff inputs:
    # diff_inputs = torch.log(diff_inputs / (1 - diff_inputs))

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

    n_neurons = calculate_neurons(th_0.shape[1])

    W1_1, W1_2 = th_0[:, 0:n_neurons].unsqueeze(1).unsqueeze(-1), th_1[:, 0:n_neurons].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, n_neurons, 1)
    b1_1, b1_2 = th_0[:, n_neurons:2*n_neurons].unsqueeze(1).unsqueeze(-1), th_1[:, n_neurons:2*n_neurons].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, n_neurons, 1)
    W2_1, W2_2 = th_0[:, 2*n_neurons:2*n_neurons + n_neurons**2].reshape((bs, n_neurons, n_neurons)), th_1[:, 2*n_neurons:2*n_neurons + n_neurons**2].reshape((bs, n_neurons, n_neurons)) 
    b2_1, b2_2 = th_0[:, 2*n_neurons + n_neurons**2:3*n_neurons + n_neurons**2], th_1[:, 2*n_neurons + n_neurons**2:3*n_neurons + n_neurons**2] 
    W3_1, W3_2 = th_0[:, 3*n_neurons + n_neurons**2:3*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, n_neurons)), th_1[:, 3*n_neurons + n_neurons**2:3*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, n_neurons)) 
    b3_1, b3_2 = th_0[:, 3*n_neurons + 2*n_neurons**2:4*n_neurons + 2*n_neurons**2], th_1[:, 3*n_neurons + 2*n_neurons**2:4*n_neurons + 2*n_neurons**2] 
    if iterated: 
        W4_1, W4_2 = th_0[:, 4*n_neurons + 2*n_neurons**2:9*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, 5)), th_1[:, 4*n_neurons + 2*n_neurons**2:9*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, 5)) 
        b4_1, b4_2 = th_0[:, 9*n_neurons + 2*n_neurons**2:9*n_neurons + 2*n_neurons**2 + 5], th_1[:, 9*n_neurons + 2*n_neurons**2:9*n_neurons + 2*n_neurons**2 + 5] 
    else:
        W4_1, W4_2 = th_0[:, 4*n_neurons + 2*n_neurons**2:5*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, 1)), th_1[:, 4*n_neurons + 2*n_neurons**2:5*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, 1)) 
        b4_1, b4_2 = th_0[:, 5*n_neurons + 2*n_neurons**2:5*n_neurons + 2*n_neurons**2 + n_neurons], th_1[:, 5*n_neurons + 2*n_neurons**2:5*n_neurons + 2*n_neurons**2 + n_neurons] 
    
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
        # clear figure
        plt.clf()
        # set y axis to be -0.05 to 1.05
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        plt.scatter(diff_inputs.squeeze(), p_1.mean(dim=0).squeeze(), label=f"{p1}")
        plt.scatter(diff_inputs.squeeze(), p_2.mean(dim=0).squeeze(), label=f"{p2}")
        plt.legend()
        # save fig to filename, account for if no folder exists
        if not os.path.exists(f"images/{name}"):
            os.makedirs(f"images/{name}")
        plt.savefig(f"images/{name}/diff_graph{torch.rand(1)}.png")
        # plt.show()
        diff_output = torch.mean(torch.abs(p_1 - p_2), dim=1)

    return diff_output


def main(th_0, th_1, upper_bound=1.0, iterated=False, p1=None, p2=None, name="test"):
    bs = th_0.shape[0]
    n_params = th_0.shape[1]
    
    def f(th, input):
        linspace = torch.linspace(0, 1, n_params).to(device)
        # Compute sigmoid of th[0]
        probs_1 = torch.sigmoid(th)
        
        # Expand input dimensions to perform operations
        expanded_input = input.unsqueeze(-1)
        
        # Compute absolute difference between input and linspace
        diffs = torch.abs(expanded_input - linspace)
        
        # Get indices of the two smallest differences
        _, indices = diffs.topk(2, dim=-1, largest=False)
        
        # Interpolate between the two closest values
        lower_indices, upper_indices = indices[..., 0].unsqueeze(-1), indices[..., 1].unsqueeze(-1)
        lower_values, upper_values = probs_1.gather(1, lower_indices), probs_1.gather(1, upper_indices)
        
        lower_diffs, upper_diffs = diffs.gather(1, lower_indices), diffs.gather(1, upper_indices)
        
        # Compute weights for interpolation based on inverse of differences
        lower_weight = 1 - lower_diffs / (lower_diffs + upper_diffs)
        upper_weight = 1 - lower_weight
        
        # Interpolate
        interpolated_values = lower_weight * lower_values + upper_weight * upper_values
        
        return interpolated_values
    
    diff_inputs = []
    p_1_values = []
    p_2_values = []
    
    for _ in range(100):
        input1 = torch.rand(1).to(device) * upper_bound
        # repeat input to make it have shape: (bs)
        input = input1.repeat(bs)
        diff_inputs.append(input1)
        p_1_values.append(f(th_0, input))
        p_2_values.append(f(th_1, input))

    # Stacking the values along a new dimension
    p_1 = torch.stack(p_1_values, dim=1)
    p_2 = torch.stack(p_2_values, dim=1)
    diff_inputs = torch.stack(diff_inputs, dim=1)

    # Convert everything to numpy for easier sorting
    diff_inputs_np = diff_inputs.squeeze().cpu().numpy()
    p_1_np = p_1.mean(dim=0).squeeze().cpu().numpy()
    p_2_np = p_2.mean(dim=0).squeeze().cpu().numpy()

    # Sort them
    sorted_indices = np.argsort(diff_inputs_np)
    diff_inputs_sorted = diff_inputs_np[sorted_indices]
    p_1_sorted = p_1_np[sorted_indices]
    p_2_sorted = p_2_np[sorted_indices]

    # Plotting
    plt.clf()
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.plot(diff_inputs_sorted, p_1_sorted, label=f"{p1}")
    plt.plot(diff_inputs_sorted, p_2_sorted, '--', label=f"{p2}")
    plt.legend()
    if not os.path.exists(f"images/{name}"):
        os.makedirs(f"images/{name}")
    plt.savefig(f"images/{name}/diff_graph{torch.rand(1)}.png")
    
    return None


if __name__ == "__main__":
    main()