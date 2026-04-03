
import json
import os
import io
import re
import time
from logging import getLogger
from collections import OrderedDict, defaultdict
from typing import List

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
import seaborn as sns
import matplotlib.pyplot as plt

from .optim import get_optimizer
from .utils import to_cuda
from const_opt import refine, compute_metrics

has_apex = True
try:
    import apex
except ImportError:
    has_apex = False

logger = getLogger()


def encode(sequences: List, env, params) -> torch.Tensor:
    float_scalar_descriptor_len = 3
    res = []
    for seq in sequences:
        seq_toks = []
        for x, y in seq:
            x_toks = env.float_encoder.encode(x)
            y_toks = env.float_encoder.encode(y)
            input_dim = int(len(x_toks) / (2 + params.mantissa_len))
            output_dim = int(len(y_toks) / (2 + params.mantissa_len))
            x_toks = [
                *x_toks,
                *[
                    "<INPUT_PAD>"
                    for _ in range(
                        (params.max_input_dimension - input_dim)
                        * float_scalar_descriptor_len
                    )
                ],
            ]
            y_toks = [
                *y_toks,
                *[
                    "<OUTPUT_PAD>"
                    for _ in range(
                        (params.max_output_dimension - output_dim)
                        * float_scalar_descriptor_len
                    )
                ],
            ]
            toks = [*x_toks, *y_toks]
            seq_toks.append([env.equation_word2id[tok] for tok in toks])
        res.append(torch.LongTensor(seq_toks))

    res = torch.stack(res)

    length = res.shape[1]
    bs = res.shape[0]
    length = torch.LongTensor(np.ones(bs) * length)

    return res, length


def ids2tokens(ids, token_list, padding_idx=82):
    pre_order_sequence = []
    for token_id in ids:
        if token_id == padding_idx:
            break
        token_word = token_list[token_id]
        pre_order_sequence.append(token_word)

    return pre_order_sequence

def gen2eq(env, params, encoded_y, generations, sample_to_learn, stored_skeletons):
    dimension = sample_to_learn['x_to_fit'][0].shape[1]
    x_gt = sample_to_learn['x_to_fit'][0].reshape(-1,dimension) 
    y_gt = sample_to_learn['y_to_fit'][0].reshape(-1,1) 
    x_gt_pred = sample_to_learn['x_to_predict'][0].reshape(-1,dimension) 
    y_gt_pred = sample_to_learn['y_to_predict'][0].reshape(-1,1) 
    generations = generations.unsqueeze(-1)
    generations = generations.transpose(1, 2).cpu().tolist()
    generations_tree = [
        list(filter(lambda x: x is not None,
                [env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                    for hyp in generations[i]
                ],))for i in range(len(encoded_y))]
    non_skeleton_tree = copy.deepcopy(generations_tree)
    skeleton_candidate, _ = env.generator.function_to_skeleton(non_skeleton_tree[0][0], constants_with_idx=False)
    if skeleton_candidate.infix() in stored_skeletons:
        return False, skeleton_candidate , None, None, None, None, None, None, None, None 
    else:
        refined = refine(env, x_gt, y_gt, non_skeleton_tree[0], verbose=True)
        best_tree = list(filter(lambda gen: gen["refinement_type"]=='BFGS', refined))
        tree = best_tree[0]['predicted_tree']

        numexpr_fn = env.simplifier.tree_to_numexpr_fn(tree) 
        y = numexpr_fn(x_gt)[:,0].reshape(-1,1)
        y_pred = numexpr_fn(x_gt_pred)[:,0].reshape(-1,1)
        if np.isnan(y).any():
            mse_fit = float('inf')
        else:
            mse_fit = np.mean((y-y_gt)**2) / ( np.mean(y_gt**2) + 1e-10)
            mse_pred = np.mean((y_pred - y_gt_pred)**2) / ( np.mean(y_gt**2) + 1e-10)

        complexity = len(tree.prefix().split(','))

        results_fit = compute_metrics(
            {
                "true": [y_gt],
                "predicted": [y],
                "predicted_tree": [tree],
            },
            metrics=params.validation_metrics,
        )
        results_predict = compute_metrics(
            {
                "true": [y_gt_pred],
                "predicted": [y_pred],
                "predicted_tree": [tree],
            },
            metrics=params.validation_metrics,
        )
        eq_outputs = (True, skeleton_candidate, tree, complexity, y, y_pred , mse_fit, mse_pred, results_fit, results_predict)
        return eq_outputs
    
def gen2eq_dir(env, params, encoded_y, generations, sample_to_learn, stored_skeletons):
    dimension = sample_to_learn['x_to_fit'][0].shape[1]
    x_gt = sample_to_learn['x_to_fit'][0].reshape(-1,dimension) 
    y_gt = sample_to_learn['y_to_fit'][0].reshape(-1,1) 
    x_gt_pred = sample_to_learn['x_to_predict'][0].reshape(-1,dimension) 
    y_gt_pred = sample_to_learn['y_to_predict'][0].reshape(-1,1) 
    generations = generations.unsqueeze(-1)
    generations = generations.transpose(1, 2).cpu().tolist()
    generations_tree = [
        list(filter(lambda x: x is not None,
                [env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                    for hyp in generations[i]
                ],))for i in range(len(encoded_y))]
    non_skeleton_tree = copy.deepcopy(generations_tree)
    skeleton_candidate, _ = env.generator.function_to_skeleton(non_skeleton_tree[0][0], constants_with_idx=False)
    if skeleton_candidate.infix() in stored_skeletons:
        return False, skeleton_candidate, None, None, None, None, None, None, None, None
    else:
        eq_outputs = (True, skeleton_candidate, generations_tree[0][0], None, None, None, None, None, None, None)
        return eq_outputs

def contrastive_loss(z_dist, y_dist, alpha=100):
    B = z_dist.shape[0]
    
    L_similar =  (alpha * z_dist - y_dist)**2

    valid_pairs = torch.triu(torch.ones(B, B), diagonal=1).bool()
    num_valid_pairs = valid_pairs.sum()
    loss = (L_similar)[valid_pairs].sum() / num_valid_pairs
    
    return loss , L_similar


def format_equation_coefficients(equation_str: str, decimal_places: int = 3) -> str:
    pattern = r'-?\d+\.\d+'
    
    def replace_number(match):
        number_str = match.group(0)
        try:
            number = float(number_str)
            rounded_number = round(number, decimal_places)
            
            if rounded_number == int(rounded_number):
                return str(int(rounded_number))
            else:
                return f"{rounded_number:.{decimal_places}f}".rstrip('0').rstrip('.')
        except ValueError:
            return number_str
    
    formatted_equation = re.sub(pattern, replace_number, equation_str)
    
    return formatted_equation

def parameterize_equation_coefficients(equation_str: str) -> str:
    coeff_idx = [0]
    def repl(match):
        idx = coeff_idx[0]
        coeff_idx[0] += 1
        return f"C{idx}"
    return re.sub(r'-?\d+\.\d+(?:[eE][+-]?\d+)?', repl, equation_str)

def scramble_encoder_y_parameters(encoder_y: nn.Module) -> None:
    with torch.no_grad():
        for name, param in encoder_y.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)
    
    logger.info("Successfully scrambled encoder_y parameters")

def generate_random_z_from_stats(encoded_y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mean_encoded_y = torch.mean(encoded_y)
        std_encoded_y = torch.std(encoded_y)
        
        std_encoded_y = torch.clamp(std_encoded_y, min=1e-6)
        
        batch_size = encoded_y.shape[0]
        random_noise = torch.randn_like(encoded_y)
        
        random_z = mean_encoded_y + std_encoded_y * random_noise
        
        logger.info(f"Generated random z with mean: {mean_encoded_y.mean().item():.6f}, "
                   f"std: {std_encoded_y.mean().item():.6f}, shape: {random_z.shape}")
        
        return random_z

class LoadParameters(object):
    def __init__(self, modules, params):
        self.modules = modules
        self.params = params
        self.set_parameters()

    def set_parameters(self):
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1


class Trainer(object):
    def __init__(self, modules, env, params, path=None, root=None):
        self.modules = modules
        self.params = params
        self.env = env

        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = self.total_samples = self.n_equations = 0
        self.infos_statistics = defaultdict(list)
        self.errors_statistics = defaultdict(int)

        self.iterators = {}

        self.set_parameters()

        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        assert not params.nvidia_apex or has_apex

        self.set_optimizer()

        self.scaler = None
        if params.amp >= 0:
            self.init_amp()

        if params.multi_gpu:
            self.wrap_ddp()

        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-np.inf if biggest else np.inf)
            for (metric, biggest) in self.metrics
        }

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [("processed_e", 0)]
            + [("processed_w", 0)]
            + [("Contrastive_Loss", [])]
            + [("Recon_Loss", [])]
            + sum(
                [[(x, []), (f"{x}-AVG-STOP-PROBS", [])] for x in env.TRAINING_TASKS], []
            )
        )
        self.last_time = time.time()
        
        self.current_vae_losses = {}

        # self.reload_checkpoint(path=path, root=root)

        if params.export_data:
            assert params.reload_data == ""
            params.export_path_prefix = os.path.join(params.dump_path, "data.prefix")
            self.file_handler_prefix = io.open(
                params.export_path_prefix, mode="a", encoding="utf-8"
            )
            logger.info(
                f"Data will be stored in prefix in: {params.export_path_prefix} ..."
            )

        if params.reload_data != "":
            logger.info(params.reload_data)
            assert params.export_data is False
            s = [x.split(",") for x in params.reload_data.split(";") if len(x) > 0]
            assert (
                len(s)
                >= 1
            )
            self.data_path = {
                task: (
                    train_path if train_path != "" else None,
                    valid_path if valid_path != "" else None,
                    test_path if test_path != "" else None,
                )
                for task, train_path, valid_path, test_path in s
            }

            logger.info(self.data_path)

            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)
        else:
            self.data_path = None

        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = {
                task: iter(self.env.create_train_iterator(task, self.data_path, params))
                for task in params.tasks
            }

    def set_new_train_iterator_params(self, args={}):
        params = self.params
        if params.env_base_seed < 0:
            params.env_base_seed = np.random.randint(1_000_000_000)
        self.dataloader = {
            task: iter(
                self.env.create_train_iterator(task, self.data_path, params, args)
            )
            for task in params.tasks
        }
        logger.info(
            "Succesfully replaced training iterator with following args:{}".format(args)
        )
        return

    def set_parameters(self):
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        params = self.params
        self.optimizer = get_optimizer(
            self.parameters["model"], params.lr, params.optimizer
        )
        logger.info("Optimizer: %s" % type(self.optimizer))

    def init_amp(self):
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )
        mod_names = sorted(self.modules.keys())
        if params.nvidia_apex is True:
            modules, optimizer = apex.amp.initialize(
                [self.modules[k] for k in mod_names],
                self.optimizer,
                opt_level=("O%i" % params.amp),
            )
            self.modules = {k: module for k, module in zip(mod_names, modules)}
            self.optimizer = optimizer
        else:
            self.scaler = torch.cuda.amp.GradScaler()

    def wrap_ddp(self):
        from torch.nn.parallel import DistributedDataParallel as DDP
        local_rank = self.params.local_rank
        trainable_keys = ["cvae", "seq_decoder", "feature_fusion"]
        # Keep unwrapped references for non-forward methods (e.g. generate_from_latent)
        self.modules_raw = {k: v for k, v in self.modules.items()}
        for k in trainable_keys:
            if k in self.modules:
                logger.info(f"Wrapping {k} with DDP on device {local_rank} ...")
                self.modules[k] = DDP(
                    self.modules[k],
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True,
                )
                logger.info(f"Wrapped {k} successfully")
        # Re-collect parameters after DDP wrapping
        self.set_parameters()
        logger.info("Wrapped modules with DDP: %s" % sorted(trainable_keys))

    def get_raw_module(self, key):
        """Get unwrapped module (for non-forward methods like generate_from_latent)."""
        if hasattr(self, 'modules_raw'):
            return self.modules_raw[key]
        return self.modules[key]

    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.warning("NaN detected")

        params = self.params

        optimizer = self.optimizer

        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
            optimizer.step()

        elif params.nvidia_apex is True:
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    clip_grad_norm_(
                        apex.amp.master_params(self.optimizer), params.clip_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(
                    loss, optimizer, delay_unscale=True
                ) as scaled_loss:
                    scaled_loss.backward()

        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
            self.scaler.scale(loss).backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

    def iter(self):
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        if self.n_total_iter % self.params.print_freq != 0:
            return

        s_total_eq = "- Total Eq: " + "{:.2e}".format(self.n_equations)
        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        
        if hasattr(self, 'current_vae_losses') and len(self.current_vae_losses) > 0:
            vae_stat = (
                f"VAE-TOTAL: {self.current_vae_losses['total_loss']:.4f} || "
                f"VAE-PRED-PRIOR: {self.current_vae_losses['pred_loss_prior']:.4f} || "
                f"VAE-PRED-POST: {self.current_vae_losses['pred_loss_post']:.4f} || "
                f"VAE-KL: {self.current_vae_losses['kl_loss']:.4f} || "
                f"VAE-KL-WEIGHT: {self.current_vae_losses['kl_weight']:.4f}"
            )
            s_stat = s_stat + " || " + vae_stat if s_stat else vae_stat
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        s_lr = (" - LR: ") + " / ".join(
            "{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups
        )

        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats["processed_e"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        max_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        s_mem = " MEM: {:.2f} MB - ".format(max_mem)
        self.stats["processed_e"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time
        logger.info(s_iter + s_speed + s_mem + s_stat + s_lr + s_total_eq)

    def get_generation_statistics(self, task):

        total_eqs = sum(
            x.shape[0]
            for x in self.infos_statistics[list(self.infos_statistics.keys())[0]]
        )
        logger.info("Generation statistics (to generate {} eqs):".format(total_eqs))

        all_infos = defaultdict(list)
        for info_type, infos in self.infos_statistics.items():
            all_infos[info_type] = torch.cat(infos).tolist()
            infos = [torch.bincount(info) for info in infos]
            max_val = max([info.shape[0] for info in infos])
            aggregated_infos = torch.cat(
                [
                    F.pad(info, (0, max_val - info.shape[0])).unsqueeze(-1)
                    for info in infos
                ],
                -1,
            ).sum(-1)
            non_zeros = aggregated_infos.nonzero(as_tuple=True)[0]
            vals = [
                (
                    non_zero.item(),
                    "{:.2e}".format(
                        (aggregated_infos[non_zero] / aggregated_infos.sum()).item()
                    ),
                )
                for non_zero in non_zeros
            ]
            logger.info("{}: {}".format(info_type, vals))
        all_infos = pd.DataFrame(all_infos)
        g = sns.PairGrid(all_infos)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        plt.savefig(
            os.path.join(self.params.dump_path, "statistics_{}.png".format(self.epoch))
        )

        str_errors = "Errors ({} eqs)\n ".format(total_eqs)
        for error_type, count in self.errors_statistics.items():
            str_errors += "{}: {}, ".format(error_type, count)
        logger.info(str_errors[:-2])
        self.errors_statistics = defaultdict(int)
        self.infos_statistics = defaultdict(list)

    def save_checkpoint(self, name, include_optimizer=True):
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        cvae_mod = self.modules['cvae'].module if hasattr(self.modules['cvae'], 'module') else self.modules['cvae']
        assert cvae_mod.global_step == self.n_total_iter

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }
        
        if hasattr(self, 'current_vae_losses'):
            data["current_vae_losses"] = self.current_vae_losses

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            # Unwrap DDP module to save compatible checkpoint
            mod = v.module if hasattr(v, 'module') else v
            data[k] = mod.state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        if path is None:
            path = "checkpoint.pth"

        if self.params.reload_checkpoint != "":
            checkpoint_path = self.params.reload_checkpoint
            assert os.path.isfile(checkpoint_path)
        else:
            if root is not None:
                checkpoint_path = os.path.join(root, path)
            else:
                checkpoint_path = os.path.join(self.params.dump_path, path)
            if not os.path.isfile(checkpoint_path):
                logger.warning(
                    "Checkpoint path does not exist, {}".format(checkpoint_path)
                )
                return

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        for k, v in self.modules.items():
            if k not in data:
                logger.warning(f"Module '{k}' not found in checkpoint, skipping.")
                continue
            weights = data[k]
            try:
                v.load_state_dict(weights)
            except RuntimeError:
                try:
                    weights = {name.partition(".")[2]: val for name, val in data[k].items()}
                    v.load_state_dict(weights)
                except RuntimeError:
                    missing, unexpected = v.load_state_dict(data[k], strict=False)
                    if missing:
                        logger.warning(f"Module '{k}': missing keys (randomly initialized): {missing}")
                    if unexpected:
                        logger.warning(f"Module '{k}': unexpected keys (ignored): {unexpected}")
            v.requires_grad = requires_grad

        logger.warning("Not reloading checkpoint optimizer.")
        if "optimizer" in data:
            for group_id, param_group in enumerate(self.optimizer.param_groups):
                if "num_updates" not in param_group:
                    logger.warning("No 'num_updates' for optimizer.")
                    continue
                logger.warning("Reloading 'num_updates' and 'lr' for optimizer.")
                param_group["num_updates"] = data["optimizer"]["param_groups"][
                    group_id
                ]["num_updates"]
                param_group["lr"] = self.optimizer.get_lr_for_step(
                    param_group["num_updates"]
                )
        else:
            logger.warning("No optimizer state in checkpoint, using fresh optimizer.")

        if self.params.fp16 and not self.params.nvidia_apex:
            if "scaler" in data:
                logger.warning("Reloading gradient scaler ...")
                self.scaler.load_state_dict(data["scaler"])
        else:
            if "scaler" in data:
                logger.warning("Scaler in checkpoint but not expected, ignoring.")

        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
        
        if "current_vae_losses" in data and hasattr(self, 'current_vae_losses'):
            self.current_vae_losses = data["current_vae_losses"]
            logger.warning("Current VAE losses restored from checkpoint")


        cvae_mod = self.modules['cvae'].module if hasattr(self.modules['cvae'], 'module') else self.modules['cvae']
        cvae_mod.global_step = self.n_total_iter
        logger.warning(f"load vae_model.global_step: {cvae_mod.global_step}")
        
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        )


    def reload_model(self, modules_to_load, path=None, requires_grad=True):
        if path is None:
            if self.params.reload_model != "":
                model_path = self.params.reload_model
                assert os.path.isfile(model_path)
        else:
            model_path = path
            assert os.path.isfile(model_path)

        logger.warning(f"Reloading pretrained model from {model_path} ...")
        
        if not torch.cuda.is_available():
            data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            data = torch.load(model_path, weights_only=False)

        for k in modules_to_load:
            v = self.modules[k]
            weights = data[k]
            try:
                weights = data[k]
                v.load_state_dict(weights)
                logger.info(f"Loaded {k} parameters from {model_path}")
            except RuntimeError:
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)

            v.requires_grad = requires_grad

    def save_periodic(self, should_save=False):
        if should_save:
            self.save_checkpoint("periodic-%i-nice" % self.epoch)
        else:
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores, prefix=None, suffix=None):
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            _metric = metric
            if prefix is not None:
                _metric = prefix + "_" + _metric
            if suffix is not None:
                _metric = _metric + "_" + suffix
            if _metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % _metric)
                continue
            factor = 1 if biggest else -1

            if metric in self.best_metrics:
                best_so_far = factor * self.best_metrics[metric]
            else:
                best_so_far = -np.inf
            if factor * scores[_metric] > best_so_far:
                self.best_metrics[metric] = scores[_metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[_metric]))
                self.save_checkpoint("best-%s" % metric)

    def end_epoch(self, scores):
        if self.stopping_criterion is not None and (
            self.params.is_master or not self.stopping_criterion[0].endswith("_mt_bleu")
        ):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self, task):
        batch, errors = next(self.dataloader[task])
        
        return batch, errors

    def export_data(self, task):
        samples, _ = self.get_batch(task)
        for info in samples["infos"]:
            samples["infos"][info] = list(map(str, samples["infos"][info].tolist()))

        def get_dictionary_slice(idx, dico):
            x = {}
            for d in dico:
                x[d] = dico[d][idx]
            return x

        def float_list_to_str_lst(lst, float_precision):
            for i in range(len(lst)):
                for j in range(len(lst[i])):
                    str_float = f"%.{float_precision}e" % lst[i][j]
                    lst[i][j] = str_float
            return lst

        processed_e = len(samples)
        for i in range(processed_e):
            outputs = {**get_dictionary_slice(i, samples["infos"])}
            x_to_fit = samples["x_to_fit"][i].tolist()
            y_to_fit = samples["y_to_fit"][i].tolist()
            outputs["x_to_fit"] = float_list_to_str_lst(
                x_to_fit, self.params.float_precision
            )
            outputs["y_to_fit"] = float_list_to_str_lst(
                y_to_fit, self.params.float_precision
            )

            outputs["tree"] = samples["tree"][i].prefix()
            outputs["skeleton_tree_encoded"] = samples["skeleton_tree_encoded"][i]

            self.file_handler_prefix.write(json.dumps(outputs) + "\n")
            self.file_handler_prefix.flush()

        self.n_equations += processed_e
        self.total_samples += self.params.batch_size
        self.stats["processed_e"] += len(samples)


    def enc_dec_vae_step(self, task):
        params = self.params
        vae_model, decoder, embedder_f, embedder_e, feature_fusion = (
            self.modules["cvae"],
            self.modules["seq_decoder"],
            self.modules["data_encoder"],
            self.modules["token_embed"],
            self.modules["feature_fusion"],
        )

        vae_model.train()
        decoder.train()
        feature_fusion.train()
        embedder_f.train()
        embedder_e.train()

        env = self.env

        samples, errors = self.get_batch(task)

        if self.params.debug_train_statistics:
            for info_type, info in samples["infos"].items():
                self.infos_statistics[info_type].append(info)
            for error_type, count in errors.items():
                self.errors_statistics[error_type] += count

        x_to_fit = samples["x_to_fit"]
        y_to_fit = samples["y_to_fit"]

        x1 = []
        for seq_id in range(len(x_to_fit)):
            x1.append([])
            for seq_l in range(len(x_to_fit[seq_id])):
                x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])

        x1, len1 = embedder_f(x1)

        if self.params.use_skeleton:
            x2, len2 = self.env.batch_equations(
                self.env.word_to_idx(
                    samples["skeleton_tree_encoded"], float_input=False
                )
            )
        else:
             x2, len2 = self.env.batch_equations(
                self.env.word_to_idx(samples["tree_encoded"], float_input=False)
            )

        x2, len2 = to_cuda(x2, len2)

        x2_e = embedder_e(x2.transpose(0, 1)).transpose(0, 1)

        prior_mu, prior_logvar, post_mu, post_logvar, kl_loss, kl_weights, kld = vae_model(x1, x2_e, len1, len2, mode="train")

        mapped_src_enc_prior = feature_fusion(prior_mu, prior_logvar)
        mapped_src_enc_post = feature_fusion(post_mu, post_logvar)

        alen = torch.arange(params.max_src_len, dtype=torch.long, device=len2.device)
        pred_mask = (alen[:, None] < len2[None] - 1)

        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        decoded_prior = decoder(
            "fwd",
            x=x2,
            lengths=len2,
            causal=True,
            src_enc=mapped_src_enc_prior,
            src_len=len1,
        )
        scores_prior, loss_pred_prior = decoder(
            "predict", tensor=decoded_prior, pred_mask=pred_mask, y=y, get_scores=False
        )

        decoded_post = decoder(
            "fwd",
            x=x2,
            lengths=len2,
            causal=True,
            src_enc=mapped_src_enc_post,
            src_len=len1,
        )
        scores_post, loss_pred_post = decoder(
            "predict", tensor=decoded_post, pred_mask=pred_mask, y=y, get_scores=False
        )

        loss = loss_pred_prior + loss_pred_post + kl_loss

        total_loss_val = loss.item()
        pred_loss_prior_val = loss_pred_prior.item()
        pred_loss_post_val = loss_pred_post.item()
        kl_loss_val = kl_loss.item()
        kl_weight_val = kl_weights.item() if torch.is_tensor(kl_weights) else kl_weights
        
        self.current_vae_losses = {
            'total_loss': total_loss_val,
            'pred_loss_prior': pred_loss_prior_val,
            'pred_loss_post': pred_loss_post_val,
            'kl_loss': kl_loss_val,
            'kl_term': torch.mean(kld).item(),
            'kl_weight': kl_weight_val,
            'post_mu_mean': post_mu.mean().item(),
            'post_logvar_mean': post_logvar.mean().item(),
            'prior_mu_mean': prior_mu.mean().item(),
            'prior_logvar_mean': prior_logvar.mean().item()
        }

        self.stats[task].append(loss.item())

        self.optimize(loss)

        self.inner_epoch += 1
        self.n_equations += len2.size(0)
        self.stats["processed_e"] += len2.size(0)
        self.stats["processed_w"] += (len2 + len2 - 2).sum().item()

        return mapped_src_enc_prior, mapped_src_enc_post, samples, loss

    def finalize_epoch_vae_losses(self):
        pass
