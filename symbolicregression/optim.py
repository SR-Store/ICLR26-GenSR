

import re
import math
import inspect

import torch
from torch import optim
from torch.optim import AdamW


class Adam(optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"] 
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"] * group["lr"], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class AdamWithWarmup(Adam):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=10000,
        warmup_init_lr=1e-7,
    ):
        super().__init__(
            params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )

        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = lr
        self.lr_step = (lr - warmup_init_lr) / warmup_updates

        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.warmup_end_lr

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


class AdamInverseSqrtWithWarmup(Adam):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=10000,
        warmup_init_lr=1e-7,
        exp_factor=0.5,
    ):
        super().__init__(
            params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )

        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        self.exp_factor = exp_factor
        self.decay_factor = warmup_end_lr * warmup_updates ** self.exp_factor

        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -self.exp_factor)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


class AdamCosineWithWarmup(Adam):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=10000,
        warmup_init_lr=1e-7,
        min_lr=1e-9,
        init_period=100000000,
        period_mult=1,
        lr_shrink=0.75,
        smooth=False,
    ):
        super().__init__(
            params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )

        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.smooth = smooth
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        self.min_lr = min_lr
        self.max_lr = lr
        self.period = init_period
        self.period_mult = period_mult
        self.lr_shrink = lr_shrink
        assert not self.smooth or self.period_mult == 1

        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            t = num_updates - self.warmup_updates
            if self.period_mult == 1:
                if self.smooth:
                    pid = math.floor(t / self.period - 1 / 2)
                else:
                    pid = math.floor(t / self.period)
                t_i = self.period
                t_curr = t - (self.period * pid)
            else:
                pid = math.floor(
                    math.log(
                        1 - t / self.period * (1 - self.period_mult), self.period_mult
                    )
                )
                t_i = self.period * (self.period_mult ** pid)
                t_curr = (
                    t
                    - (1 - self.period_mult ** pid)
                    / (1 - self.period_mult)
                    * self.period
                )
            lr_shrink = self.lr_shrink ** pid
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink
            if self.smooth:
                return min_lr + 0.5 * (max_lr - min_lr) * (
                    1 + math.cos(2 * math.pi * t_curr / t_i)
                )
            else:
                return min_lr + 0.5 * (max_lr - min_lr) * (
                    1 + math.cos(math.pi * t_curr / t_i)
                )

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


def get_optimizer(parameters, lr, s):
    if "," in s:
        method = s[: s.find(",")]
        optim_params = {}
        for x in s[s.find(",") + 1 :].split(","):
            split = x.split("=")
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == "adadelta":
        optim_fn = optim.Adadelta
    elif method == "adagrad":
        optim_fn = optim.Adagrad
    elif method == "adam":
        optim_fn = Adam
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_warmup":
        optim_fn = AdamWithWarmup
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_inverse_sqrt":
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_cosine":
        optim_fn = AdamCosineWithWarmup
        optim_params["smooth"] = False
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_smooth_cosine":
        optim_fn = AdamCosineWithWarmup
        optim_params["smooth"] = True
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adamax":
        optim_fn = optim.Adamax
    elif method == "asgd":
        optim_fn = optim.ASGD
    elif method == "rmsprop":
        optim_fn = optim.RMSprop
    elif method == "rprop":
        optim_fn = optim.Rprop
    elif method == "sgd":
        optim_fn = optim.SGD
        assert "lr" in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception(
            'Unexpected parameters: expected "%s", got "%s"'
            % (str(expected_args[2:]), str(optim_params.keys()))
        )

    return optim_fn(parameters, lr=lr, **optim_params)

