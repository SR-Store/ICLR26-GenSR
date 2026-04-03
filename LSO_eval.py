import torch
import numpy as np
import pandas as pd
import sympy as sp
import os, sys
import symbolicregression
import wandb

from symbolicregression.envs import build_env
from symbolicregression.model import check_model_params, build_modules
from parsers import get_parser
from symbolicregression.trainer_vae import Trainer

from collections import OrderedDict, defaultdict
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
from symbolicregression.model.model_wrapper import ModelWrapper
import symbolicregression.model.utils_wrapper as utils_wrapper

from symbolicregression.metrics import compute_metrics
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from sklearn.model_selection import train_test_split
from pathlib import Path
from model import VAESymbolicRegressor
from LSO_fit import lso_fit_es_covfromvae_fit

import time
from tqdm import tqdm
import copy


import re

def extract_info(path):
    model_match = re.search(r'run-train-([^/]+)/', path)
    period_match = re.search(r'periodic-(\d+)(?:-[^.]*)?\.pth', path)
    model = model_match.group(1) if model_match else None
    period = period_match.group(1) if period_match else None
    return model, period


def reload_model(modules, path, requires_grad=False):
    if path is None:
        path = "checkpoint.pth"
    assert os.path.isfile(path)

    data = torch.load(path, weights_only=False)

    print(f"load model {path} successful")
    if 'epoch' in data:
        print(f"train epoch: {data['epoch']}")
    if 'params' in data:
        print(f"train dump_path: {data['params'].get('dump_path', 'N/A')}")

    for k, v in modules.items():
        if k not in data:
            if k == "feature_fusion" and "latent_bridge" in data:
                print(f"Module 'feature_fusion': loading from legacy key 'latent_bridge'")
                weights = data["latent_bridge"]
            elif k == "feature_fusion" and "mapper" in data:
                print(f"Module 'feature_fusion': loading from legacy key 'mapper'")
                weights = data["mapper"]
            else:
                continue
        else:
            weights = data[k]
        try:
            v.load_state_dict(weights, strict=False)
            missing = set(v.state_dict().keys()) - set(weights.keys())
            if missing:
                print(f"Module '{k}': missing keys (randomly initialized): {sorted(missing)}")
        except RuntimeError:
            stripped = {name[len("module."):] if name.startswith("module.") else name: val for name, val in weights.items()}
            try:
                v.load_state_dict(stripped, strict=False)
                missing = set(v.state_dict().keys()) - set(stripped.keys())
                if missing:
                    print(f"Module '{k}': missing keys after strip (randomly initialized): {sorted(missing)}")
            except RuntimeError as e2:
                print(f"Warning: Could not load module '{k}': {e2}")
                continue
        print(f"load model {k} successful")
        for p in v.parameters():
            p.requires_grad_(requires_grad)

    return modules


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


def evaluate_pmlb_lso(
        trainer,
        params,
        target_noise=0.0,
        random_state=29910,
        verbose=False,
        save=True,
        filter_fn=None,
        logger=None,
        save_file=None,
        save_suffix="./eval_result/eval_pmlb_lso.csv",
        rescale = True):

        env = trainer.env
        params = params

        if params.reload_model != "" and params.reload_model_dir != "":
            path = os.path.join(params.reload_model_dir, params.reload_model)
            trainer.modules = reload_model(trainer.modules, path)

        model = VAESymbolicRegressor(params = params, env=env, modules=trainer.modules)

        model.to(params.device)
        all_datasets = pd.read_csv(
            "./datasets/pmlb/pmlb/all_summary_stats.tsv",
            sep="\t",)
        regression_datasets = all_datasets[all_datasets["task"] == "regression"]
        regression_datasets = regression_datasets[
            regression_datasets["n_categorical_features"] == 0]
        problems = regression_datasets

        if filter_fn is not None:
            problems = problems[filter_fn(problems)]
            problems = problems.loc[problems['n_features']<11]
        problem_names = problems["dataset"].values.tolist()

        pmlb_path = "./datasets/pmlb/datasets/"

        feynman_problems = pd.read_csv(
            "./datasets/feynman/FeynmanEquations.csv",
            delimiter=",",)
        feynman_problems = feynman_problems[["Filename", "Formula"]].dropna().values
        feynman_formulas = {}
        for p in range(feynman_problems.shape[0]):
            feynman_formulas[
                "feynman_" + feynman_problems[p][0].replace(".", "_")
            ] = feynman_problems[p][1]
        if save:
            save_file = save_suffix


        if params.filter_feynman_subset and params.pmlb_data_type=="feynman":
            problem_names = []
            problem_names.append("feynman_III_14_14")
            problem_names.append("feynman_III_15_12")
        elif params.sweep_problems == 1:
            problem_names = []
            problem_names.append("feynman_III_8_54")
            problem_names.append("feynman_test_13")
            problem_names.append("feynman_III_10_19")
            problem_names.append("feynman_III_12_43")
            problem_names.append("feynman_III_14_14")
            problem_names.append("feynman_III_15_12")
        elif params.feynman_sel_equs_num != -1:
            problem_names = problem_names[:params.feynman_sel_equs_num]

        rng = np.random.RandomState(random_state)
        pbar = tqdm(total=len(problem_names))
        first_write = True
        counter = 0

        aggregate_metrics = {
            'r2_scores': [],
            'r2_scores_fit': [],
            'complexities': [],
            'runtimes': [],
            'iteration_counts': [],
            'avg_iteration_times': [],
            'success_rate': 0,
            'total_problems': len(problem_names)
        }
        params.unfinished = 0

        if params.reverse_hard_probelms != -1:
            valid_idx = len(problem_names) - params.reverse_hard_probelms

        for idx, problem_name in enumerate(problem_names):
            counter += 1
            batch_results = defaultdict(list)

            if params.reverse_hard_probelms != -1:
                if idx <= valid_idx:
                    pbar.update(1)
                    continue

            # No overall early stopping — always run all equations by default

            print("Sample: ", counter)
            if problem_name in feynman_formulas:
                formula = feynman_formulas[problem_name]
            else:
                formula = "???"
            print("GT equation : ", formula)
            print("EQ: ", problem_name)

            X, y, _ = read_file(pmlb_path + "{}/{}.tsv.gz".format(problem_name, problem_name))
            y = np.expand_dims(y, -1)

            x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=random_state)

            scale = target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
            noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
            y_to_fit += noise

            if not isinstance(X, list):
                X = [x_to_fit]
                Y = [y_to_fit]

            scaler = utils_wrapper.StandardScaler() if rescale else None
            scale_params = {}
            if scaler is not None:
                scaled_X = []
                for i, x in enumerate(X):
                    scaled_X.append(scaler.fit_transform(x))
                    scale_params[i]=scaler.get_params()
            else:
                scaled_X = X

            bag_number =1
            done_bagging = False
            bagging_threshold = 0.99
            max_r2_zero = 0
            max_bags = 1
            while (done_bagging == False) and (bag_number <= max_bags):
                bag_number += 1
                X_scaled_to_fit = scaled_X[0]
                Y_scaled_to_fit = Y[0]

                sample_to_learn = {'X_scaled_to_fit': 0, 'Y_scaled_to_fit':0, 'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
                sample_to_learn['X_scaled_to_fit'] = [X_scaled_to_fit]
                sample_to_learn['Y_scaled_to_fit'] = [Y_scaled_to_fit]
                sample_to_learn['x_to_fit'] = [x_to_fit]
                sample_to_learn['y_to_fit'] = [y_to_fit]
                sample_to_learn['x_to_predict'] = [x_to_predict]
                sample_to_learn['y_to_predict'] = [y_to_predict]

                sample_to_learn['eq_gt'] = [formula]

                with torch.no_grad():
                    batch_results = lso_fit_es_covfromvae_fit(sample_to_learn, env, params, model, batch_results, bag_number, es_strategy=params.es_strategy)

                current_iterations = len(batch_results["all_iteration_times"])

                batch_results_filtered = {k: v for k, v in batch_results.items() if k != "all_iteration_times"}
                batch_results = pd.DataFrame.from_dict(batch_results_filtered)
                batch_results.insert(0, "problem", problem_name)
                batch_results.insert(0, "formula", formula)
                batch_results["input_dimension"] = x_to_fit.shape[1]
                batch_results["bag_number"] = bag_number
                max_r2_zero = batch_results["r2_zero_final_fit"][0]

                print("R2 zero final fit: ", batch_results["r2_zero_final_fit"][0])

                current_r2 = 0.0
                current_runtime = 0.0
                current_complexity = 0

                current_r2_fit = 0.0
                try:
                    if "r2_final_predict" in batch_results and len(batch_results["r2_final_predict"]) > 0:
                        current_r2 = max(0, batch_results["r2_final_predict"][0])

                    if "r2_final_fit" in batch_results and len(batch_results["r2_final_fit"]) > 0:
                        current_r2_fit = max(0, batch_results["r2_final_fit"][0])

                    if "time" in batch_results and len(batch_results["time"]) > 0:
                        current_runtime = batch_results["time"][0]

                    if "_complexity_final_predict" in batch_results:
                        current_complexity = batch_results["_complexity_final_predict"][0]

                except Exception as e:
                    print(f"Warning: Error extracting metrics for {problem_name}: {e}")
                    current_r2 = 0.0
                    current_runtime = 0.0
                    current_complexity = 0

                current_avg_iter_time = current_runtime / max(current_iterations, 1) if current_runtime > 0 and current_iterations > 0 else 0.0

                aggregate_metrics['r2_scores'].append(current_r2)
                aggregate_metrics['r2_scores_fit'].append(current_r2_fit)
                aggregate_metrics['complexities'].append(current_complexity)
                aggregate_metrics['runtimes'].append(current_runtime)
                aggregate_metrics['iteration_counts'].append(current_iterations)
                aggregate_metrics['avg_iteration_times'].append(current_avg_iter_time)

            if save:
                dir_name = os.path.dirname(save_file)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                if first_write:
                    batch_results.to_csv(save_file, index=False)
                    first_write = False
                else:
                    batch_results.to_csv(
                        save_file, mode="a", header=False, index=False
                    )
            pbar.update(1)

            r2_eq_1_num = sum(r2 == 1.0 for r2 in aggregate_metrics['r2_scores'])
            r2_eq_1_ratio = r2_eq_1_num / len(aggregate_metrics['r2_scores'])

            wandb.log({
                "current/r2_score": current_r2,
                "current/complexity": current_complexity,
                "current/runtime_seconds": current_runtime,
                "current/iterations": current_iterations,
                "mean/r2_score": np.mean(aggregate_metrics['r2_scores']),
                "mean/complexity": np.mean(aggregate_metrics['complexities']),
                "mean/runtime_seconds": np.mean(aggregate_metrics['runtimes']),
                "mean/iterations": np.mean(aggregate_metrics['iteration_counts']),
                "mean/avg_iteration_time": np.mean(aggregate_metrics['avg_iteration_times']),
                "mean/r2_eq_1_ratio": r2_eq_1_ratio,
                "mean/r2_eq_1_num": r2_eq_1_num
            }, step=counter)

        if aggregate_metrics['r2_scores']:
            success_count = sum(1 for r2 in aggregate_metrics['r2_scores'] if r2 > 0.99)
            success_rate = success_count / len(aggregate_metrics['r2_scores'])

            final_aggregate_metrics = {
                "aggregate/avg_r2_score": np.mean(aggregate_metrics['r2_scores']),
                "aggregate/median_r2_score": np.median(aggregate_metrics['r2_scores']),
                "aggregate/std_r2_score": np.std(aggregate_metrics['r2_scores']),
                "aggregate/avg_complexity": np.mean(aggregate_metrics['complexities']),
                "aggregate/median_complexity": np.median(aggregate_metrics['complexities']),
                "aggregate/std_complexity": np.std(aggregate_metrics['complexities']),
                "aggregate/avg_runtime": np.mean(aggregate_metrics['runtimes']),
                "aggregate/median_runtime": np.median(aggregate_metrics['runtimes']),
                "aggregate/total_runtime": np.sum(aggregate_metrics['runtimes']),
                "aggregate/avg_iterations": np.mean(aggregate_metrics['iteration_counts']),
                "aggregate/avg_iteration_time": np.mean(aggregate_metrics['avg_iteration_times']),
                "aggregate/success_rate": success_rate,
                "aggregate/success_count": success_count,
                "aggregate/total_problems": len(aggregate_metrics['r2_scores']),
                "aggregate/best_r2": np.max(aggregate_metrics['r2_scores']),
                "aggregate/worst_r2": np.min(aggregate_metrics['r2_scores'])
            }

            summary_table = wandb.Table(
                columns=["Metric", "Mean", "Median", "Std", "Min", "Max"],
                data=[
                    ["R2 Score", np.mean(aggregate_metrics['r2_scores']),
                     np.median(aggregate_metrics['r2_scores']),
                     np.std(aggregate_metrics['r2_scores']),
                     np.min(aggregate_metrics['r2_scores']),
                     np.max(aggregate_metrics['r2_scores'])],
                    ["Complexity", np.mean(aggregate_metrics['complexities']),
                     np.median(aggregate_metrics['complexities']),
                     np.std(aggregate_metrics['complexities']),
                     np.min(aggregate_metrics['complexities']),
                     np.max(aggregate_metrics['complexities'])],
                    ["Runtime (s)", np.mean(aggregate_metrics['runtimes']),
                     np.median(aggregate_metrics['runtimes']),
                     np.std(aggregate_metrics['runtimes']),
                     np.min(aggregate_metrics['runtimes']),
                     np.max(aggregate_metrics['runtimes'])],
                    ["Avg Iter Time (s)", np.mean(aggregate_metrics['avg_iteration_times']),
                     np.median(aggregate_metrics['avg_iteration_times']),
                     np.std(aggregate_metrics['avg_iteration_times']),
                     np.min(aggregate_metrics['avg_iteration_times']),
                     np.max(aggregate_metrics['avg_iteration_times'])]
                ]
            )
            wandb.log({"summary_statistics": summary_table}, step=len(aggregate_metrics['r2_scores']))

            equations_table = wandb.Table(
                columns=["Problem", "Formula", "R2", "Complexity", "Runtime", "Success"],
                data=[[problem_names[i],
                      "GT_formula" if i < len(problem_names) else "Unknown",
                      aggregate_metrics['r2_scores'][i],
                      aggregate_metrics['complexities'][i],
                      aggregate_metrics['runtimes'][i],
                      "Y" if aggregate_metrics['r2_scores'][i] > 0.99 else "N"]
                     for i in range(len(aggregate_metrics['r2_scores']))]
            )
            wandb.log({"detailed_results": equations_table}, step=len(aggregate_metrics['r2_scores']))

            print(f"\nAggregate Results Summary:")
            print(f"Average R2 Score (predict): {np.mean(aggregate_metrics['r2_scores']):.4f}")
            print(f"Average R2 Score (fit): {np.mean(aggregate_metrics['r2_scores_fit']):.4f}")
            print(f"Success Rate (R2 > 0.99): {success_rate:.2%}")
            print(f"Average Complexity: {np.mean(aggregate_metrics['complexities']):.2f}")
            print(f"Average Runtime: {np.mean(aggregate_metrics['runtimes']):.2f}s")
            print(f"Average Iteration Time: {np.mean(aggregate_metrics['avg_iteration_times']):.4f}s")

            wandb.log({"final/mean_r2_predict": np.mean(aggregate_metrics['r2_scores']),
                       "final/mean_r2_fit": np.mean(aggregate_metrics['r2_scores_fit']),
                       "final/mean_complexity": np.mean(aggregate_metrics['complexities']),
                       "final/mean_runtime": np.mean(aggregate_metrics['runtimes']),
                       "final/mean_avg_iteration_time": np.mean(aggregate_metrics['avg_iteration_times']),
                       "final/mean_iteration_counts": np.mean(aggregate_metrics['iteration_counts']),
                       "final/unfinished": params.unfinished,
                       "final/equations_processed": len(aggregate_metrics['r2_scores']),
                       "final/total_equations": len(problem_names)})

        pbar.close()


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    use_sample_pop, use_y_noise_pop, use_latent_noise_pop = params.pop_init_index.split('*')
    params.use_sample_pop = int(use_sample_pop)
    params.use_y_noise_pop = int(use_y_noise_pop)
    params.use_latent_noise_pop = int(use_latent_noise_pop)

    params.batch_size = 1
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    params.n_steps_per_epoch = 100
    params.max_input_dimension = 10
    params.env_base_seed = 2023
    params.n_dec_layers = 16

    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1
    params.random_state = 14423
    params.max_number_bags = 10
    params.eval_verbose_print = True
    params.rescale = True
    params.n_trees_to_refine = params.beam_size

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    if params.eval_in_train_mode:
        modules = build_modules(env, params, mode='train')
    else:
        modules = build_modules(env, params, mode='eval')

    trainer = Trainer(modules, env, params)

    if params.eval_lso_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type
        save = params.save_results

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman"))

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

        params.train_info, params.train_period = extract_info(params.reload_model)

        import datetime
        import uuid
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_string = uuid.uuid4().hex[:6]
        suf = f'{current_time}_{random_string}'

        setting = 'fe{}_{}'.format(
            params.feynman_sel_equs_num,
            suf)

        wandb_group_name = getattr(params, 'wandb_group_name', None) or f'{group_name}'
        wandb_run_name = getattr(params, 'wandb_run_name', None) or f'eval-{params.pmlb_data_type}-seed{params.seed}-{current_time}'
        wandb_project = getattr(params, 'wandb_project', None) or "symbolic-regression-lso"

        if params.wandb_disabled:
            run = wandb.init(mode="disabled")
        elif getattr(params, 'wandb_resume_id', ''):
            wandb.init(
                project=wandb_project,
                id=params.wandb_resume_id,
                resume="must",
                config=params,
            )
        else:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                group=wandb_group_name,
                config=params,
            )

        save_dir = f"./eval_result/noise/res_{wandb_group_name}/{wandb_run_name}.csv"

        if os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        evaluate_pmlb_lso(
            trainer,
            params,
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            save_file=None,
            save_suffix=save_dir
        )

        wandb.finish()

