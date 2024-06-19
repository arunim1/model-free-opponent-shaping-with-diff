import torch
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# TODO: make the piecewise linear one-shot game


def asymmetrize(p_m_1, p_m_2, eps=1e-3):
    raise NotImplementedError

    # implementation below not yet tested - likely incorrect.
    p_m_1 = p_m_1.clone()  # needed because of in-place operations + device apparently
    p_m_2 = p_m_2.clone()
    # [(2,2), (0,3+e), (3,0), (1,1+e)], i.e. changing player 2's incentive to defect
    p_m_1 += torch.tensor([[0, eps], [0, eps]]).to(
        device
    )  # correct if p_m_1 corresponds to player 2's loss
    p_m_2 += torch.tensor([[0, 0], [0, 0]]).to(device)
    return p_m_1, p_m_2


# Original max_abs_diff function
def max_abs_diff(arr1, arr2):
    diffs = torch.abs(arr1 - arr2)
    return torch.max(diffs, dim=1).values


def mellowmax(arr1, arr2, alpha=50):
    diffs = torch.abs(arr1 - arr2)
    return (1 / alpha) * torch.log(torch.mean(torch.exp(alpha * diffs), dim=1))


def one_shot(payout_mat_1, payout_mat_2, bs, asym=None, threshold=None, pwlinear=None):
    dims = [1, 1] if pwlinear is None else [pwlinear, pwlinear]
    # implement some version of asymetry calculation here.
    if asym is not None:
        payout_mat_1, payout_mat_2 = asymmetrize(payout_mat_1, payout_mat_2, asym)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)

    def Ls(th):
        # th is List[Tensor(bs, 1), Tensor(bs,1)], representing the inputs from each player.
        # the th[0] corresponds to: p_1, payout_mat_1, L_1
        # the th[1] corresponds to: p_2, payout_mat_2, L_2

        if threshold is not None:
            t_1, t_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
            # t_1, t_2 are each player's selected threshold, between 0 and 1.
            # player 1 chooses action 1 if diff + noise < t_1, and action 2 if diff + noise >= t_1. since noise is uniform between 0 and 1, this becomes:
            # P(player 1 action 1) = thresh1 - diff

            if threshold == "exp0":
                # n\left(a,b\right)\ =\ \operatorname{abs}\left(\frac{\left(e^{a}-1\right)}{e-1}-\frac{\left(e^{b}-1\right)}{e-1}\right)
                diff = torch.abs(torch.exp(t_1) - torch.exp(t_2)) / (torch.e - 1)
            elif threshold == "exp1":
                # m\left(a,b\right)\ =\frac{\left(e^{\left(\left(\operatorname{abs}\left(a-b\right)\right)\right)}-1\right)}{e-1}
                diff = (torch.exp(torch.abs(t_1 - t_2)) - 1) / (torch.e - 1)
            elif threshold == "squared":
                diff = (t_1 - t_2) ** 2
                # second derivative is always -2 or some constant
            elif threshold == "quartic":
                diff = (t_1 - t_2) ** 4
                # second derivative is variable
            else:
                diff = torch.abs(t_1 - t_2)
                # second derivative is always 0

            p_1 = torch.relu(t_1 - diff)
            p_2 = torch.relu(t_2 - diff)

        elif pwlinear is not None:
            assert th[0].shape[1] == th[1].shape[1] == pwlinear

            y_1, y_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
            y_1 = y_1 * (1 - 1e-5)  # preventing diffs of 1
            y_2 = y_2 * (1 - 1e-5)
            # here, y_1 and y_2 are arrays of shape (bs, pwlinear)

            # diffs = max_abs_diff(y_1, y_2)
            diffs = mellowmax(y_1, y_2)
            assert torch.all(diffs >= 0) and torch.all(diffs <= 1)

            y_idx = torch.floor(diffs * (pwlinear - 1))
            interval = 1 / (pwlinear - 1)

            p_1_minus = y_1[range(bs), y_idx.int()]
            p_1_plus = y_1[range(bs), y_idx.int() + 1]
            p_2_minus = y_2[range(bs), y_idx.int()]
            p_2_plus = y_2[range(bs), y_idx.int() + 1]

            w_minus = diffs - (y_idx * interval)
            w_plus = ((1 + y_idx) * interval) - diffs

            # p is got by interpolating between the values closes to diff in y
            p_1 = ((p_1_minus * w_minus + p_1_plus * w_plus) / interval).unsqueeze(1)
            p_2 = ((p_2_minus * w_minus + p_2_plus * w_plus) / interval).unsqueeze(1)

        else:
            # non-diff-meta game
            p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])

        assert torch.all(p_1 >= 0) and torch.all(
            p_2 >= 0
        ), f"Negative probability found: p_1={p_1}, p_2={p_2}"
        assert torch.all(p_1 <= 1) and torch.all(
            p_2 <= 1
        ), f"Probability greater than 1 found: p_1={p_1}, p_2={p_2}"

        x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
        M = torch.bmm(x.view(bs, 2, 1), y.view(bs, 1, 2)).view(bs, 1, 4)
        # M is the probability of each outcome (CC, CD, DC, DD) at the present step, where C is action 1 and D is action 2, and CD means p_1 chooses action 1 and p_2 chooses action 2.
        # unfortunately, p_1 (which corresponds to payoff_mat_1) is p2 in the outer loop for non-MFOS at least.
        L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
        L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]

    return dims, Ls


def iterated(payout_mat_1, payout_mat_2, bs, gamma_inner=0.96, asym=None, ccdr=None):
    raise NotImplementedError  # mostly implemented, but shouldn't be used here!
    dims = [5, 5]
    if asym is not None:
        payout_mat_1, payout_mat_2 = asymmetrize(payout_mat_1, payout_mat_2, asym)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)

    def Ls(
        th,
    ):  # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = torch.sigmoid(th[0][:, 0:1])
        p_2_0 = torch.sigmoid(th[1][:, 0:1])
        p = torch.cat(
            [
                p_1_0 * p_2_0,
                p_1_0 * (1 - p_2_0),
                (1 - p_1_0) * p_2_0,
                (1 - p_1_0) * (1 - p_2_0),
            ],
            dim=-1,
        )
        p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = torch.reshape(
            torch.sigmoid(
                torch.cat(
                    [th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1
                )
            ),
            (bs, 4, 1),
        )
        P = torch.cat(
            [p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1
        )

        M = torch.matmul(
            p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P)
        )
        L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]

    return dims, Ls


def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True, allow_unused=True)[0]
    return grad


class MetaGames:
    def __init__(
        self,
        b,
        pms,
        opponent="NL",
        lr=None,
        asym=None,
        threshold=None,
        pwlinear=None,
        seed=None,
        ccdr=None,
        adam=False,
    ):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn.
        COPYCAT = Copies what opponent played last step.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.gamma_inner = 0.96
        self.b = b
        self.ccdr = ccdr
        assert self.ccdr is None
        self.adam = adam

        self.p1 = "MFOS"
        self.p2 = opponent
        self.opponent = opponent

        # assuming one-shot games
        self.payoff_mat_1 = pms[0]
        self.payoff_mat_2 = pms[1]

        self.lr = 1 if lr is None else lr
        self.asym = asym
        self.threshold = threshold
        self.pwlinear = pwlinear

        assert self.pwlinear is None or self.threshold is None

        d, self.game_batched = one_shot(
            self.payoff_mat_1,
            self.payoff_mat_2,
            self.b,
            asym=self.asym,
            threshold=self.threshold,
            pwlinear=self.pwlinear,
        )

        self.std = 1 if self.pwlinear is None else 0.05
        self.d = d[0]

        self.init_th_ba = None

    def reset(self, info=False):
        if self.init_th_ba is not None:
            self.inner_th_ba = self.init_th_ba.detach() * torch.ones(
                (self.b, self.d), requires_grad=True
            ).to(device)
        else:
            self.inner_th_ba = torch.nn.init.normal_(
                torch.empty((self.b, self.d), requires_grad=True), std=self.std
            ).to(device)
        outer_th_ba = torch.nn.init.normal_(
            torch.empty((self.b, self.d), requires_grad=True), std=self.std
        ).to(device)

        self.timestep = 0
        if self.adam:
            self.beta1 = 0.99
            self.beta2 = 0.999  # try adjusting this
            self.eps = 1e-8

            self.p1_m = torch.zeros_like(outer_th_ba)
            self.p1_v = torch.zeros_like(outer_th_ba)
            self.p2_m = torch.zeros_like(self.inner_th_ba)
            self.p2_v = torch.zeros_like(self.inner_th_ba)

        state, _, _, M = self.step(outer_th_ba)
        if info:
            return state, M
        else:
            return state

    def step(self, outer_th_ba):
        last_inner_th_ba = self.inner_th_ba.detach().clone()
        if self.opponent == "NL":
            th_ba = [self.inner_th_ba, outer_th_ba.detach()]
            l1, l2, M = self.game_batched(th_ba)
            grad = get_gradient(l1.sum(), self.inner_th_ba)
            if self.adam:
                self.p1_m = self.beta1 * self.p1_m + (1 - self.beta1) * grad
                self.p1_v = self.beta2 * self.p1_v + (1 - self.beta2) * (grad**2)
                m_hat = self.p1_m / (1 - self.beta1 ** (self.timestep + 1))
                v_hat = self.p1_v / (1 - self.beta2 ** (self.timestep + 1))
                grad = m_hat / (torch.sqrt(v_hat) + self.eps)

            with torch.no_grad():
                self.inner_th_ba -= grad * self.lr
        elif self.opponent == "LOLA":
            th_ba = [self.inner_th_ba, outer_th_ba.detach()]
            th_ba[1].requires_grad = True
            l1, l2, M = self.game_batched(th_ba)
            losses = [l1, l2]
            grad_L = [
                [get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)]
                for i in range(2)
            ]
            term = (grad_L[1][0] * grad_L[1][1]).sum()
            grad = grad_L[0][0] - self.lr * get_gradient(term, th_ba[0])

            if self.adam:
                self.p1_m = self.beta1 * self.p1_m + (1 - self.beta1) * grad
                self.p1_v = self.beta2 * self.p1_v + (1 - self.beta2) * (grad**2)
                m_hat = self.p1_m / (1 - self.beta1 ** (self.timestep + 1))
                v_hat = self.p1_v / (1 - self.beta2 ** (self.timestep + 1))
                grad = m_hat / (torch.sqrt(v_hat) + self.eps)

            with torch.no_grad():
                self.inner_th_ba -= grad * self.lr
        elif self.opponent == "BR":
            num_steps = 1000
            inner_th_ba = torch.nn.init.normal_(
                torch.empty((self.b, 5), requires_grad=True), std=self.std
            ).to(device)
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

        self.timestep += 1

        return (
            [
                torch.sigmoid(outer_th_ba).detach(),
                torch.sigmoid(last_inner_th_ba).detach(),
            ],
            -l2.detach(),
            -l1.detach(),
            M,
        )


class SymmetricMetaGames:
    def __init__(self, b, game="IPD"):
        self.gamma_inner = 0.96

        self.b = b
        self.game = game
        if self.game == "IPD":
            d, self.game_batched = ipd_batched(b, gamma_inner=self.gamma_inner)
            self.std = 1
        elif self.game == "IMP":
            d, self.game_batched = imp_batched(b, gamma_inner=self.gamma_inner)
            self.std = 1
        elif self.game == "chicken":
            d, self.game_batched = chicken_game_batch(b)
            self.std = 1
        else:
            raise NotImplementedError

        self.d = d[0]

    def reset(self, info=False):
        p_ba_0 = torch.nn.init.normal_(torch.empty((self.b, self.d)), std=self.std).to(
            device
        )
        p_ba_1 = torch.nn.init.normal_(torch.empty((self.b, self.d)), std=self.std).to(
            device
        )
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

        if self.game == "IPD" or self.game == "IMP":
            return (
                [state_0, state_1],
                [-l1 * (1 - self.gamma_inner), -l2 * (1 - self.gamma_inner)],
                M,
            )
        else:
            return [state_0, state_1], [-l1.detach(), -l2.detach()], M


class NonMfosMetaGames:
    def __init__(
        self,
        b,
        pms,
        p1="NL",
        p2="NL",
        lr=None,
        asym=None,
        threshold=None,
        pwlinear=None,
        seed=None,
        ccdr=None,
        adam=False,
    ):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn. Used for sanity checking.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.gamma_inner = 0.96
        self.b = b
        self.ccdr = ccdr
        assert self.ccdr is None
        self.adam = adam

        self.p1 = p1
        self.p2 = p2

        # assuming one-shot games
        self.payoff_mat_1 = pms[0]
        self.payoff_mat_2 = pms[1]

        self.lr = 1 if lr is None else lr
        self.asym = asym
        self.threshold = threshold
        self.pwlinear = pwlinear

        assert self.pwlinear is None or self.threshold is None

        d, self.game_batched = one_shot(
            self.payoff_mat_1,
            self.payoff_mat_2,
            self.b,
            asym=self.asym,
            threshold=self.threshold,
            pwlinear=self.pwlinear,
        )

        self.std = 1 if self.pwlinear is None else 0.05
        self.d = d[0]

        self.init_th_ba = None

    def reset(self, info=False):
        self.p1_th_ba = torch.nn.init.normal_(
            torch.empty((self.b, self.d), requires_grad=True), std=self.std
        ).to(device)
        self.p2_th_ba = torch.nn.init.normal_(
            torch.empty((self.b, self.d), requires_grad=True), std=self.std
        ).to(device)

        self.timestep = 0

        if self.adam:
            self.beta1 = 0.99
            self.beta2 = 0.999  # try adjusting this
            self.eps = 1e-8

            self.p1_m = torch.zeros_like(self.p1_th_ba)
            self.p1_v = torch.zeros_like(self.p1_th_ba)
            self.p2_m = torch.zeros_like(self.p2_th_ba)
            self.p2_v = torch.zeros_like(self.p2_th_ba)

        state, _, _, M = self.step()
        if info:
            return state, M

        return None

    def step(self, info=False):
        last_p1_th_ba = self.p1_th_ba.clone()
        last_p2_th_ba = self.p2_th_ba.clone()
        th_ba = [self.p2_th_ba, self.p1_th_ba]
        l1, l2, M = self.game_batched(th_ba)

        # UPDATE P1
        if self.p1 == "NL":
            grad = get_gradient(l2.sum(), self.p1_th_ba)
            if self.adam:
                self.p1_m = self.beta1 * self.p1_m + (1 - self.beta1) * grad
                self.p1_v = self.beta2 * self.p1_v + (1 - self.beta2) * (grad**2)
                m_hat = self.p1_m / (1 - self.beta1 ** (self.timestep + 1))
                v_hat = self.p1_v / (1 - self.beta2 ** (self.timestep + 1))
                grad = m_hat / (torch.sqrt(v_hat) + self.eps)
            with torch.no_grad():
                self.p1_th_ba -= grad * self.lr
        elif self.p1 == "LOLA":
            losses = [l1, l2]
            grad_L = [
                [get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)]
                for i in range(2)
            ]
            term = (grad_L[0][0] * grad_L[0][1]).sum()
            grad = grad_L[1][1] - self.lr * get_gradient(term, th_ba[1])
            if self.adam:
                self.p1_m = self.beta1 * self.p1_m + (1 - self.beta1) * grad
                self.p1_v = self.beta2 * self.p1_v + (1 - self.beta2) * (grad**2)
                m_hat = self.p1_m / (1 - self.beta1 ** (self.timestep + 1))
                v_hat = self.p1_v / (1 - self.beta2 ** (self.timestep + 1))
                grad = m_hat / (torch.sqrt(v_hat) + self.eps)
            with torch.no_grad():
                self.p1_th_ba -= grad * self.lr
        elif self.p1 == "STATIC":
            pass
        else:
            raise NotImplementedError

        # UPDATE P2
        if self.p2 == "NL":
            grad = get_gradient(l1.sum(), self.p2_th_ba)
            if self.adam:
                self.p1_m = self.beta1 * self.p1_m + (1 - self.beta1) * grad
                self.p1_v = self.beta2 * self.p1_v + (1 - self.beta2) * (grad**2)
                m_hat = self.p1_m / (1 - self.beta1 ** (self.timestep + 1))
                v_hat = self.p1_v / (1 - self.beta2 ** (self.timestep + 1))
                grad = m_hat / (torch.sqrt(v_hat) + self.eps)
            with torch.no_grad():
                self.p2_th_ba -= grad * self.lr
        elif self.p2 == "LOLA":
            losses = [l1, l2]
            grad_L = [
                [get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)]
                for i in range(2)
            ]
            term = (grad_L[1][0] * grad_L[1][1]).sum()
            grad = grad_L[0][0] - self.lr * get_gradient(term, th_ba[0])
            if self.adam:
                self.p1_m = self.beta1 * self.p1_m + (1 - self.beta1) * grad
                self.p1_v = self.beta2 * self.p1_v + (1 - self.beta2) * (grad**2)
                m_hat = self.p1_m / (1 - self.beta1 ** (self.timestep + 1))
                v_hat = self.p1_v / (1 - self.beta2 ** (self.timestep + 1))
                grad = m_hat / (torch.sqrt(v_hat) + self.eps)
            with torch.no_grad():
                self.p2_th_ba -= grad * self.lr
        elif self.p2 == "STATIC":
            pass
        else:
            raise NotImplementedError

        self.timestep += 1

        return (
            [torch.sigmoid(last_p1_th_ba), torch.sigmoid(last_p2_th_ba)],
            -l2.detach(),
            -l1.detach(),
            M,
        )
