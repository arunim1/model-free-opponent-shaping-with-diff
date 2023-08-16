import torch

# debugging

import matplotlib.pyplot as plt
import torch 
import math
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore



def main(th_0, th_1, upper_bound=1.0, iterated=False, p1=None, p2=None, name="test", graph=False):
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

    # desired outputs: inputs passed through \frac{\left(\cos\left(5x\right)+1\right)}{2}
    desired_outputs = ((torch.cos(5 * diff_inputs_repeated) + 1) / 2).squeeze(-1)

    def calculate_neurons(n_params):
        a = 2
        b = 9 if iterated else 5
        c = -(n_params - 5) if iterated else -(n_params - 1)
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        neuron1 = (-b + math.sqrt(discriminant)) / (2*a)
        neuron2 = (-b - math.sqrt(discriminant)) / (2*a)
        return math.floor(max(neuron1, neuron2))
    
    n_neurons = calculate_neurons(th_0.shape[1])

    W1_1, W1_2 = th_0[:, 0:n_neurons].unsqueeze(1).unsqueeze(-1), th_1[:, 0:n_neurons].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, n_neurons, 1)
    b1_1, b1_2 = th_0[:, n_neurons:2*n_neurons].unsqueeze(1).unsqueeze(-1), th_1[:, n_neurons:2*n_neurons].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, n_neurons, 1)
    W2_1, W2_2 = th_0[:, 2*n_neurons:2*n_neurons + n_neurons**2].reshape((bs, n_neurons, n_neurons)), th_1[:, 2*n_neurons:2*n_neurons + n_neurons**2].reshape((bs, n_neurons, n_neurons)) 
    b2_1, b2_2 = th_0[:, 2*n_neurons + n_neurons**2:3*n_neurons + n_neurons**2], th_1[:, 2*n_neurons + n_neurons**2:3*n_neurons + n_neurons**2] 
    W3_1, W3_2 = th_0[:, 3*n_neurons + n_neurons**2:3*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, n_neurons)), th_1[:, 3*n_neurons + n_neurons**2:3*n_neurons + 2*n_neurons**2].reshape((bs, n_neurons, n_neurons)) 
    b3_1, b3_2 = th_0[:, 3*n_neurons + 2*n_neurons**2:4*n_neurons + 2*n_neurons**2], th_1[:, 3*n_neurons + 2*n_neurons**2:4*n_neurons + 2*n_neurons**2] 
    
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

    loss_1 = torch.mean(torch.abs(p_1 - desired_outputs), dim=(0,1,2)).squeeze()
    loss_2 = torch.mean(torch.abs(p_2 - desired_outputs), dim=(0,1,2)).squeeze()

    batch_loss_1 = torch.mean(torch.abs(p_1 - desired_outputs), dim=(1,2)).squeeze()
    batch_loss_2 = torch.mean(torch.abs(p_2 - desired_outputs), dim=(1,2)).squeeze()

    def get_gradient(loss, th):
        return torch.autograd.grad(loss, th, retain_graph=True)[0]
    
    # grad_1 = get_gradient(loss_1, th_0) * bs #* th_0.shape[1] / 100
    # grad_2 = get_gradient(loss_2, th_1) * bs #* th_1.shape[1] / 100


    def get_batch_gradient(batch_loss, th):
        identity = torch.eye(batch_loss.size(0)).diag().to(batch_loss.device)
        # Compute the gradient for each sample in the batch
        grad, = torch.autograd.grad(batch_loss, th, grad_outputs=identity, retain_graph=True)
        
        return grad
    print(batch_loss_1.shape)
    print(th_0.shape)
    grad_1 = get_batch_gradient(batch_loss_1, th_0) 
    grad_2 = get_batch_gradient(batch_loss_2, th_1) 


    with torch.no_grad():
        th_0 = th_0 - grad_1
        th_1 = th_1 - grad_2

        if graph:
            # clear figure
            plt.clf()
            plt.scatter(diff_inputs.squeeze(), p_1.mean(dim=0).squeeze(), label=f"{p1}")
            plt.scatter(diff_inputs.squeeze(), p_2.mean(dim=0).squeeze(), label=f"{p2}")
            plt.scatter(diff_inputs.squeeze(), desired_outputs.mean(dim=0).squeeze(), label="desired")
            plt.legend()
            #save fig to filename, account for if no folder exists
            if not os.path.exists(f"images/{name}"):
                os.makedirs(f"images/{name}")
            plt.savefig(f"images/{name}/diff_graph{torch.rand(1)}.png")
            # diff_output = torch.mean(torch.abs(p_1 - p_2), dim=1)

    return th_0.to(device), th_1.to(device), loss_1.detach(), loss_2.detach()


if __name__ == "__main__":
    n_neurons_in = 4
    num_steps = 10000
    b = 200
    d = 2*(n_neurons_in**2) + 5*n_neurons_in + 1 

    list_of_tensors = []
    for _ in range(b):
        tensor = torch.nn.init.xavier_normal_(torch.empty(1, d, requires_grad=True))
        list_of_tensors.append(tensor)
    p1_th_ba = torch.cat(list_of_tensors, dim=0).to(device)

    list_of_tensors = []
    for _ in range(b):
        tensor = torch.nn.init.xavier_normal_(torch.empty(1, d, requires_grad=True))
        list_of_tensors.append(tensor)
    p2_th_ba = torch.cat(list_of_tensors, dim=0).to(device)

    pbar = tqdm(range(num_steps), desc="Training")
    for i in pbar: 
        p1_th_ba, p2_th_ba, l1, l2 = main(p1_th_ba, p2_th_ba, iterated=False, p1="NL", p2="NL", graph=(i==num_steps-1))
        # make sure p1_th_ba, p2_th_ba require grad
        p1_th_ba.requires_grad = True
        p2_th_ba.requires_grad = True
        pbar.set_description(f"Loss: {l1:.2f}, {l2:.2f}")