import os.path as osp
import torch

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


def diff_nn(th_0, th_1):
    # output = torch.sum(torch.abs(th_0 - th_1), dim=-1, keepdim=True)
    # output = torch.norm(th_0 - th_1, dim=-1, keepdim=True)
    """
    we start by simply saying that the diff value *will be* in the range [0, 1]
    this means each policy th_0, th_1 can be said to map a value in [0, 1] to a value in [0, 1], where the output is a p(cooperate)
    or for iterated games, mapping a value in [0, 1] to a 5-vector with each entry in [0, 1] 
    """

    bs = th_0.shape[0]

    # Generate an array of inputs within the range [0, 1].
    diff_inputs = torch.linspace(0, 0.2, 100).unsqueeze(1).unsqueeze(1).to(device)

    # Repeat diff_inputs to make it have shape: (bs, 100, 1, 1)
    diff_inputs_repeated = diff_inputs.repeat(bs, 1, 1, 1)
    if th_0.shape[1] == 7: # one-shot diff game
        # First layer
        W1_1, W1_2 = th_0[:, 0:2].unsqueeze(1).unsqueeze(-1), th_1[:, 0:2].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, 2, 1)
        b1_1, b1_2 = th_0[:, 2:4].unsqueeze(1).unsqueeze(-1), th_1[:, 2:4].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, 2, 1)

        # Second layer
        W2_1, W2_2 = th_0[:, 4:6].reshape((bs, 2, 1)), th_1[:, 4:6].reshape((bs, 2, 1))
        b2_1, b2_2 = th_0[:, 6:7], th_1[:, 6:7]  # each has size (bs, 1)
    elif th_0.shape[1] == 40: # iterated diff game
        # First layer
        W1_1, W1_2 = th_0[:, 0:5].unsqueeze(1).unsqueeze(-1), th_1[:, 0:5].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, 5, 1)
        b1_1, b1_2 = th_0[:, 5:10].unsqueeze(1).unsqueeze(-1), th_1[:, 5:10].unsqueeze(1).unsqueeze(-1)  # has size (bs, 1, 5, 1)
        W2_1, W2_2 = th_0[:, 10:35].reshape((bs, 5, 5)), th_1[:, 10:35].reshape((bs, 5, 5)) # has length 25
        b2_1, b2_2 = th_0[:, 35:40], th_1[:, 35:40] # has length 5

    x1_1, x1_2 = torch.relu(diff_inputs_repeated * W1_1 + b1_1), torch.relu(diff_inputs_repeated * W1_2 + b1_2)  # each has size (bs, 100, 2, 1)

    b2_1_u, b2_2_u = b2_1.unsqueeze(1), b2_2.unsqueeze(1) 
    x1_1_u, x1_2_u = x1_1.squeeze(-1), x1_2.squeeze(-1)  # each has size (bs, 1, 100, 2)

    # Note: Using torch.matmul here instead of torch.bmm
    x2_1, x2_2 = torch.matmul(x1_1_u, W2_1) + b2_1_u, torch.matmul(x1_2_u, W2_2) + b2_2_u

    p_1, p_2 = torch.sigmoid(x2_1), torch.sigmoid(x2_2)

    return torch.mean(torch.abs(p_1 - p_2), dim=1)


def def_Ls_NN(p_m_1, p_m_2, bs, gamma_inner=0.96, iterated=False, diff_game=False):
    def Ls(th): # th is a list of two tensors, each of shape (bs, 40) for iterated games 
        th[0] = th[0].clone().to(device) # really not quite sure why this is necessary but it is. 
        th[1] = th[1].clone().to(device)
        if iterated:
            if diff_game:
                diff = diff_nn(th[0], th[1]) # has shape (bs, 1)

                diff1 = diff + 0.1 * torch.rand_like(diff)
                diff2 = diff + 0.1 * torch.rand_like(diff)

                # first layer
                W1_1, W1_2 = th[0][:, 0:5], th[1][:, 0:5] # has length 5
                b1_1, b1_2 = th[0][:, 5:10], th[1][:, 5:10] # has length 5
                x1_1, x1_2 = torch.relu(diff1 * W1_1 + b1_1), torch.relu(diff2 * W1_2 + b1_2)
                # second layer goes from 5 to 5, meaning each matrix is 5x5, and there are 5 outputs so 5 biases
                W2_1, W2_2 = th[0][:, 10:35].reshape((bs, 5, 5)), th[1][:, 10:35].reshape((bs, 5, 5)) # has length 25
                b2_1, b2_2 = th[0][:, 35:40], th[1][:, 35:40] # has length 5
                b2_1_u, b2_2_u = b2_1.unsqueeze(1), b2_2.unsqueeze(1)
                x1_1, x1_2 = x1_1.unsqueeze(1), x1_2.unsqueeze(1)
                x2_1, x2_2 = torch.bmm(x1_1, W2_1) + b2_1_u, torch.bmm(x1_2, W2_2) + b2_2_u
                x2_1, x2_2 = x2_1.squeeze(1), x2_2.squeeze(1) # outputs of NN

                p_1_0, p_2_0 = torch.sigmoid(x2_1[:, 0:1]), torch.sigmoid(x2_2[:, 0:1])
                p_1 = torch.reshape(torch.sigmoid(x2_1[:, 1:5]), (bs, 4, 1))
                p_2 = torch.reshape(torch.sigmoid(torch.cat([x2_2[:, 1:2], x2_2[:, 3:4], x2_2[:, 2:3], x2_2[:, 4:5]], dim=-1)), (bs, 4, 1))
            else:
                p_1_0, p_2_0 = torch.sigmoid(th[0][:, 0:1]), torch.sigmoid(th[1][:, 0:1])
                p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
                p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))

            p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
            P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)
            M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))  
            L_1 = -torch.matmul(M, torch.reshape(p_m_1, (bs, 4, 1))) # player 2's loss, since p2's params are th[0]
            L_2 = -torch.matmul(M, torch.reshape(p_m_2, (bs, 4, 1))) # player 1's loss, since p1's params are th[1]
            return [L_1.squeeze(-1), L_2.squeeze(-1), M] 
        
        else:
            if diff_game:
                diff = diff_nn(th[0], th[1]) # has shape (bs, 1) 

                # first layer
                W1_1, W1_2 = th[0][:, 0:2], th[1][:, 0:2] # has length 2
                b1_1, b1_2 = th[0][:, 2:4], th[1][:, 2:4] # has length 2
                
                diff1 = diff + 0.1 * torch.rand_like(diff)
                diff2 = diff + 0.1 * torch.rand_like(diff)

                # second layer
                W2_1, W2_2 = th[0][:, 4:6].reshape((bs, 2, 1)), th[1][:, 4:6].reshape((bs, 2, 1))
                b2_1, b2_2 = th[0][:, 6:7], th[1][:, 6:7] # each has size (bs, 1)
                x1_1, x1_2 = torch.relu((diff1) * W1_1 + b1_1), torch.relu((diff2) * W1_2 + b1_2)
                b2_1_u, b2_2_u = b2_1.unsqueeze(1), b2_2.unsqueeze(1)
                # x1_1_u, x1_2_u = x1_1.unsqueeze(1), x1_2.unsqueeze(1) # each has size (bs, 1, 2)
                x1_1_u = x1_1.unsqueeze(1)
                x1_2_u = x1_2.unsqueeze(1)
                x2_1, x2_2 = torch.bmm(x1_1_u, W2_1) + b2_1_u, torch.bmm(x1_2_u, W2_2) + b2_2_u
                x2_1_u, x2_2_u = x2_1.squeeze(1), x2_2.squeeze(1) # outputs of NN

                p_1, p_2 = torch.sigmoid(x2_1_u), torch.sigmoid(x2_2_u)
            else:
                p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])

            x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
            M = torch.bmm(x.view(bs, 2, 1), y.view(bs, 1, 2)).view(bs, 1, 4)
            L_1 = -torch.matmul(M, torch.reshape(p_m_1, (bs, 4, 1))) # same outputs as old version, but with M
            L_2 = -torch.matmul(M, torch.reshape(p_m_2, (bs, 4, 1))) 
            # L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), p_m_1), y.unsqueeze(-1))
            # L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), p_m_2), y.unsqueeze(-1))

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

        if nn_game and self.diff_game: self.d = 40 if self.iterated else 7

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
            self.inner_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
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

        if nn_game and self.diff_game: self.d = 40 if self.iterated else 7


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
    def __init__(self, b, p1="NL", p2="NL", game="IPD", lr=None, mmapg_id=None, asym=None, nn_game=False, ccdr=False):
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
        
        global def_Ls 
        def_Ls = def_Ls_NN if nn_game else def_Ls_threshold_game

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
        else:
            raise NotImplementedError

        self.std = 1
        if lr is not None:
            self.lr = lr
        self.d = d[0]
        
        if nn_game and self.diff_game: self.d = 40 if self.iterated else 7

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
        self.p1_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        self.p2_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)

        if self.p1 == "MAMAML":
            self.p1_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)
        if self.p2 == "MAMAML":
            self.p2_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)

        state, _, _, M = self.step()
        if info:
            return state, M

        return None

    def step(self, info=False):
        last_p1_th_ba = self.p1_th_ba.clone()
        last_p2_th_ba = self.p2_th_ba.clone()
        th_ba = [self.p2_th_ba, self.p1_th_ba]
        th_CC1 = [self.p1_th_ba, self.p1_th_ba]
        th_CC2 = [self.p2_th_ba, self.p2_th_ba]
        th_DR1 = [self.p2_th_ba, torch.randn_like(self.p1_th_ba)]
        th_DR2 = [torch.randn_like(self.p2_th_ba), self.p1_th_ba]

        l1_reg, l2_reg, M = self.game_batched(th_ba) # l2 is for p1 aka p1_th_ba, l1 is for p2 aka p2_th_ba
        
        if self.ccdr: 
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
            with torch.no_grad():
                self.p1_th_ba -= grad * self.lr
        elif self.p1 == "LOLA":
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)]
            term = (grad_L[0][0] * grad_L[0][1]).sum()
            grad = grad_L[1][1] - self.lr * get_gradient(term, th_ba[1])
            with torch.no_grad():
                self.p1_th_ba -= grad * self.lr
        elif self.p1 == "STATIC":
            pass
        else:
            raise NotImplementedError
        
        # UPDATE P2
        if self.p2 == "NL" or self.p2 == "MAMAML":
            grad = get_gradient(l1.sum(), self.p2_th_ba)
            with torch.no_grad():
                self.p2_th_ba -= grad * self.lr
        elif self.p2 == "LOLA":
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)] 
            term = (grad_L[1][0] * grad_L[1][1]).sum()
            grad = grad_L[0][0] - self.lr * get_gradient(term, th_ba[0])
            with torch.no_grad():
                self.p2_th_ba -= grad * self.lr
        elif self.p2 == "STATIC":
            pass
        else:
            raise NotImplementedError

        if self.iterated:
            return torch.sigmoid(torch.cat([last_p1_th_ba, last_p2_th_ba])), -l2 * (1 - self.gamma_inner), -l1 * (1 - self.gamma_inner), M
        else:
            return torch.sigmoid(torch.cat([last_p1_th_ba, last_p2_th_ba])), -l2.detach(), -l1.detach(), M
