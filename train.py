

import json
import random
import argparse
import numpy as np
import torch
import os
import pickle
from pathlib import Path
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
import copy
import wandb

import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from parsers import get_parser

from model import VAESymbolicRegressor
from LSO_fit import lso_fit_es_covfromvae_fit
from symbolicregression.model.model_wrapper import ModelWrapper
import symbolicregression.model.utils_wrapper as utils_wrapper

np.seterr(all="raise")


def read_file(filename, label="target", sep=None):
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


def log_iteration_to_wandb(trainer, task_loss=None):
    lr = trainer.optimizer.param_groups[0]['lr']

    log_dict = {
        "iteration": trainer.n_total_iter,
        "epoch": trainer.epoch,
        "learning_rate": lr,
    }

    if task_loss is not None:
        log_dict["train/task_loss"] = task_loss

    if hasattr(trainer, 'current_vae_losses'):
        for loss_name, loss_value in trainer.current_vae_losses.items():
            if loss_name in ['post_mu_mean', 'post_logvar_mean', 'prior_mu_mean', 'prior_logvar_mean']:
                log_dict[f"train/latent_{loss_name}"] = loss_value
            else:
                log_dict[f"train/vae_{loss_name}"] = loss_value

    wandb.log(log_dict, step=trainer.n_total_iter)


def epoch_evaluation(trainer, params, logger, epoch, eval_threshold=0.99):
    env = trainer.env

    model = VAESymbolicRegressor(params=params, env=env, modules=trainer.modules)
    model.to(params.device)

    eval_problems = ["feynman_III_8_54", "feynman_I_50_26"]
    pmlb_path = "./datasets/pmlb/datasets/"

    r2_scores = []
    random_state = 29910

    logger.info(f"Starting epoch {epoch} evaluation on {len(eval_problems)} Feynman problems...")

    for problem_name in eval_problems:
        try:
            X, y, _ = read_file(pmlb_path + "{}/{}.tsv.gz".format(problem_name, problem_name))
            y = np.expand_dims(y, -1)

            x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=random_state)

            if not isinstance(X, list):
                X_list = [x_to_fit]
                Y_list = [y_to_fit]

            scaler = utils_wrapper.StandardScaler() if params.rescale else None
            if scaler is not None:
                scaled_X = []
                for i, x in enumerate(X_list):
                    scaled_X.append(scaler.fit_transform(x))
            else:
                scaled_X = X_list

            sample_to_learn = {
                'X_scaled_to_fit': scaled_X,
                'Y_scaled_to_fit': Y_list,
                'x_to_fit': [x_to_fit],
                'y_to_fit': [y_to_fit],
                'x_to_predict': [x_to_predict],
                'y_to_predict': [y_to_predict],
                'eq_gt': ["unknown"]
            }

            batch_results = defaultdict(list)
            bag_number = 1

            with torch.no_grad():
                batch_results = lso_fit_es_covfromvae_fit(
                    sample_to_learn, env, params, model, batch_results, bag_number,
                    es_strategy=params.es_strategy)

            if "r2_final_predict" in batch_results and len(batch_results["r2_final_predict"]) > 0:
                r2_score = max(0, batch_results["r2_final_predict"][0])
            else:
                r2_score = 0.0

            r2_scores.append(r2_score)
            logger.info(f"Problem {problem_name}: R2 = {r2_score:.4f}")

        except Exception as e:
            logger.warning(f"Error evaluating {problem_name}: {e}")
            r2_scores.append(0.0)

    avg_r2 = np.mean(r2_scores) if r2_scores else 0.0
    logger.info(f"Epoch {epoch} evaluation - Average R2: {avg_r2:.4f}, Threshold: {eval_threshold}")

    wandb.log({
        "evaluation/avg_r2": avg_r2,
        "evaluation/threshold_met": 1 if avg_r2 >= eval_threshold else 0,
        "evaluation/individual_r2": r2_scores,
    }, step=trainer.n_total_iter)

    return avg_r2 >= eval_threshold


def main(params):

    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    if not params.cpu:
        params.device = 'cuda'
        assert torch.cuda.is_available()
    else:
        params.device = 'cpu'
    symbolicregression.utils.CUDA = not params.cpu

    # Adjust batch size per GPU for DDP
    if params.multi_gpu:
        assert params.batch_size % params.world_size == 0, \
            f"batch_size ({params.batch_size}) must be divisible by world_size ({params.world_size})"
        params.batch_size = params.batch_size // params.world_size

    if not hasattr(params, 'lso_optimizer'):
        params.lso_optimizer = "es_fromvae"
    if not hasattr(params, 'lso_max_iteration'):
        params.lso_max_iteration = 50
    if not hasattr(params, 'lso_stop_r2'):
        params.lso_stop_r2 = 0.99
    if not hasattr(params, 'beam_size'):
        params.beam_size = 2
    if not hasattr(params, 'rescale'):
        params.rescale = True
    if not hasattr(params, 'validation_metrics'):
        params.validation_metrics = "r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity"

    if not hasattr(params, 'wandb_project'):
        params.wandb_project = "symbolic-regression-training"

    group_name = '{}_{}_be{}_it{}_st{}_pn{}_mn{}_s{}_n{}'.format(
        params.lso_optimizer,
        params.model_type,
        params.beam_size,
        params.lso_max_iteration,
        params.lso_stop_r2,
        params.pop_num,
        params.mu_num,
        params.ev_sigma,
        params.target_noise,
    )

    if params.wandb_disabled or not params.is_master:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=params.wandb_project,
            name=params.wandb_run_name,
            group=params.wandb_group_name,
            config=params,
        )

    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "evals_all"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)
    env = build_env(params)

    modules = build_modules(env, params)
    from symbolicregression.trainer_vae import Trainer
    trainer = Trainer(modules, env, params)

    if params.reload_data != "":
        data_types = [
            "valid{}".format(i) for i in range(1, len(trainer.data_path["functions"]))
        ]
    else:
        data_types = ["valid1"]

    trainer.n_equations = 0

    eval_threshold = getattr(params, 'eval_threshold', 0.9)
    begin_epoch = trainer.epoch

    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.inner_epoch = 0
        if params.test_flag:
            trainer.n_steps_per_epoch = 5
            trainer.params.print_freq = 1

        while trainer.inner_epoch < trainer.n_steps_per_epoch:

            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                else:
                    _, _, samples, loss = trainer.enc_dec_vae_step(task)

                    log_iteration_to_wandb(trainer, task_loss=loss.item() if hasattr(loss, 'item') else loss)

                trainer.iter()

        if params.debug_train_statistics:
            for task in params.tasks:
                trainer.get_generation_statistics(task)

        should_save = False

        trainer.epoch += 1

        if should_save or (params.save_periodic > 0 and trainer.epoch % params.save_periodic == 0) or trainer.epoch <= begin_epoch + 5:
            trainer.save_periodic(should_save)

        logger.info("============ End of epoch %i ============" % trainer.epoch)

    wandb.finish()


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()

    check_model_params(params)

    main(params)

