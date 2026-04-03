import os
import copy
import time
import random
import traceback
import multiprocessing as mp

import numpy as np
import torch

from const_opt import refine, compute_metrics
from cma_es_modular import DiagonalCMAES, DiagonalCMAES_compre


# ── Multiprocessing worker globals (inherited via fork) ──────────────
_mp_env = None
_mp_params = None
_mp_sample = None


def _mp_eval_one(args):
    """BFGS"""
    idx, gen_data, skeleton_snapshot, com_weight = args

    gen_tensor = torch.tensor(gen_data)
    dummy_latent = torch.zeros(1, 1)

    try:
        eq_outputs = gen2eq(_mp_env, _mp_params, dummy_latent, gen_tensor,
                            _mp_sample, skeleton_snapshot)
        (success, skeleton_candidate, tree, complexity, y, y_pred,
         mse_fit, mse_pred, results_fit, results_predict) = eq_outputs

        if success:
            r2 = results_fit['r2_zero'][0]
            fitness = r2 if r2 < 0.5 else r2 - complexity / com_weight
            return idx, {
                'success': True,
                'r2': r2,
                'fitness': fitness,
                'tree': tree,
                'complexity': complexity,
                'mse_fit': mse_fit,
                'mse_pred': mse_pred,
                'results_fit': results_fit,
                'results_predict': results_predict,
                'skeleton_key': skeleton_candidate.infix(),
            }
    except Exception:
        pass
    return idx, {'success': False, 'r2': 0, 'fitness': 0}


def reload_model(modules, path, requires_grad=False):
    if path is None:
        path = "checkpoint.pth"
    assert os.path.isfile(path)

    data = torch.load(path, weights_only=False)

    for k, v in modules.items():
        if k not in data:
            if k == "feature_fusion" and "latent_bridge" in data:
                print(f"Module 'feature_fusion': loading from legacy key 'latent_bridge'")
                raw_weights = data["latent_bridge"]
            elif k == "feature_fusion" and "mapper" in data:
                print(f"Module 'feature_fusion': loading from legacy key 'mapper'")
                raw_weights = data["mapper"]
            else:
                continue
        else:
            raw_weights = data[k]

        try:
            v.load_state_dict(raw_weights)
            print(f"load model {k} successful")
        except RuntimeError:
            stripped = {name[len("module."):] if name.startswith("module.") else name: w
                        for name, w in raw_weights.items()}
            v.load_state_dict(stripped)
            print(f"load model {k} successful (stripped prefix)")

        for p in v.parameters():
            p.requires_grad_(requires_grad)

    return modules


def gen2eq(env, params, encoded_y, generations, sample_to_learn, stored_skeletons):
    dimension = sample_to_learn['x_to_fit'][0].shape[1]
    x_gt = sample_to_learn['x_to_fit'][0].reshape(-1, dimension)
    y_gt = sample_to_learn['y_to_fit'][0].reshape(-1, 1)
    x_gt_pred = sample_to_learn['x_to_predict'][0].reshape(-1, dimension)
    y_gt_pred = sample_to_learn['y_to_predict'][0].reshape(-1, 1)
    generations = generations.unsqueeze(-1)
    generations = generations.transpose(1, 2).cpu().tolist()
    generations_tree = [
        list(filter(lambda x: x is not None,
                [env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                    for hyp in generations[i]
                ],)) for i in range(len(encoded_y))]

    non_skeleton_tree = copy.deepcopy(generations_tree)
    if not non_skeleton_tree or not non_skeleton_tree[0]:
        return False, None, None, None, None, None, None, None, None, None
    expr = non_skeleton_tree[0][0]
    skeleton_candidate, _ = env.generator.function_to_skeleton(
        non_skeleton_tree[0][0], constants_with_idx=False
    )
    if skeleton_candidate.infix() in stored_skeletons:
        return False, skeleton_candidate, None, None, None, None, None, None, None, None
    else:
        refined = refine(env, x_gt, y_gt, non_skeleton_tree[0], verbose=True)
        best_tree = list(filter(lambda gen: gen["refinement_type"] == 'BFGS', refined))
        if not best_tree:
            best_tree = list(filter(lambda gen: gen["refinement_type"] in ["NoRef", "NONE"], refined))
        if not best_tree:
            return False, skeleton_candidate, None, None, None, None, None, None, None, None
        tree = best_tree[0]['predicted_tree']

        numexpr_fn = env.simplifier.tree_to_numexpr_fn(tree)
        y = numexpr_fn(x_gt)[:, 0].reshape(-1, 1)
        y_pred = numexpr_fn(x_gt_pred)[:, 0].reshape(-1, 1)
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(y_pred)):
            mse_fit = float('inf')
            mse_pred = float('inf')
        else:
            y = np.clip(y, -1e150, 1e150)
            y_pred = np.clip(y_pred, -1e150, 1e150)
            mse_fit = np.mean((y - y_gt)**2) / (np.mean(y_gt**2) + 1e-10)
            mse_pred = np.mean((y_pred - y_gt_pred)**2) / (np.mean(y_gt**2) + 1e-10)

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
        eq_outputs = (True, skeleton_candidate, tree, complexity, y, y_pred,
                      mse_fit, mse_pred, results_fit, results_predict)
        return eq_outputs


def create_evo_population_covfromvae(env, params, model, sample_to_learn,
                                        encoded_y, initial_prior_logvar):
    encoded_pop = encoded_y
    all_prior_logvars = [initial_prior_logvar]

    params.pop_num_each = params.pop_num // (
        params.use_sample_pop + params.use_y_noise_pop + params.use_latent_noise_pop
    )

    X_orig = sample_to_learn['X_scaled_to_fit'][0]
    Y_orig = sample_to_learn['Y_scaled_to_fit'][0]
    seq_len = len(X_orig)
    all_indices = list(range(seq_len))
    n_sample = params.max_input_points if seq_len > params.max_input_points else seq_len // 2

    if params.use_sample_pop == 1:
        batch_X, batch_Y = [], []
        for i in range(params.pop_num_each):
            idx = random.sample(all_indices, n_sample)
            batch_X.append(X_orig[idx])
            batch_Y.append(Y_orig[idx])

        encoded_batch, batch_logvar = model.encode_only(
            {'X_scaled_to_fit': batch_X, 'Y_scaled_to_fit': batch_Y}
        )
        encoded_pop = torch.cat((encoded_pop, encoded_batch), dim=0)
        for j in range(params.pop_num_each):
            all_prior_logvars.append(batch_logvar[j:j+1])

    if params.use_y_noise_pop == 1:
        batch_X, batch_Y = [], []
        for i in range(params.pop_num_each):
            sigma = 0.01 * (i / 5 + 1)
            noised_Y = Y_orig + np.random.normal(
                0, sigma * np.std(Y_orig), Y_orig.shape
            )
            idx = random.sample(all_indices, n_sample)
            batch_X.append(X_orig[idx])
            batch_Y.append(noised_Y[idx])

        encoded_batch, batch_logvar = model.encode_only(
            {'X_scaled_to_fit': batch_X, 'Y_scaled_to_fit': batch_Y}
        )
        encoded_pop = torch.cat((encoded_pop, encoded_batch), dim=0)
        for j in range(params.pop_num_each):
            all_prior_logvars.append(batch_logvar[j:j+1])

    if params.use_latent_noise_pop == 1:
        noise = torch.randn(params.pop_num_each, *encoded_y.shape[1:]).to(params.device)
        scales = (2 * torch.rand(params.pop_num_each, 1)).to(params.device)
        encoded_batch = encoded_y + scales * noise
        encoded_pop = torch.cat((encoded_pop, encoded_batch), dim=0)
        for i in range(params.pop_num_each):
            all_prior_logvars.append(initial_prior_logvar)

    return encoded_pop, all_prior_logvars


def lso_fit_es_covfromvae_fit(sample_to_learn, env, params, model,
                              batch_results, bag_number,
                              es_strategy="adaptive_gaussian"):

    def calculate_fitness(r2, complexity):
        if r2 < 0.5:
            return r2
        else:
            complexity_penalty = complexity / params.com_weight
            return r2 - complexity_penalty

    def update_global_best(result, global_state):
        updated = False

        if result['fitness'] > global_state['max_fitness']:
            global_state['max_fitness'] = result['fitness']
            global_state['min_mse'] = result['mse_fit']
            global_state['best_eq'] = result['tree']
            global_state['best_r2'] = result['r2']
            global_state['best_complexity'] = result['complexity']
            updated = True

        return updated

    def evaluate_candidate(latent, generation, sample_to_learn, stored_skeletons):
        try:
            eq_outputs = gen2eq(env, params, latent, generation,
                                sample_to_learn, stored_skeletons)
            (success, skeleton_candidate, tree, complexity, y, y_pred,
             mse_fit, mse_pred, results_fit, results_predict) = eq_outputs

            if success:
                r2 = results_fit['r2_zero'][0]
                fitness = calculate_fitness(r2, complexity)
                skeleton_key = skeleton_candidate.infix()

                return {
                    'success': True,
                    'r2': r2,
                    'fitness': fitness,
                    'tree': tree,
                    'complexity': complexity,
                    'y': y,
                    'y_pred': y_pred,
                    'mse_fit': mse_fit,
                    'mse_pred': mse_pred,
                    'results_fit': results_fit,
                    'results_predict': results_predict,
                    'skeleton_key': skeleton_key,
                }
        except Exception as _e:
            pass

        return {'success': False, 'r2': 0, 'fitness': 0}

    def _evaluate_candidates(candidates, candidate_generations,
                              sample_to_learn, skeleton_set,
                              num_workers=1):
        n = len(candidates)
        skeleton_snapshot = list(skeleton_set)

        tasks = []
        for i in range(n):
            for b in range(params.beam_size):
                gen_data = candidate_generations[
                    params.beam_size * i + b
                ].reshape(1, -1).cpu().tolist()
                tasks.append((i, gen_data, skeleton_snapshot, params.com_weight))

        if num_workers <= 1:
            all_results = [_mp_eval_one(t) for t in tasks]
        else:
            all_results = _pool.map(_mp_eval_one, tasks)

        fitness_out = torch.zeros(n, device=params.device)
        r2_out = torch.zeros(n, device=params.device)
        new_skeletons = []
        gen_false_num = 0

        for idx, result in all_results:
            if result['success']:
                if result['fitness'] > fitness_out[idx].item():
                    fitness_out[idx] = result['fitness']
                    r2_out[idx] = result['r2']
                update_global_best(result, global_best)
                if 'skeleton_key' in result:
                    new_skeletons.append(result['skeleton_key'])
            else:
                gen_false_num += 1

        skeleton_set.update(new_skeletons)
        return fitness_out, r2_out, gen_false_num

    sub_sample = copy.deepcopy(sample_to_learn)

    seq_len = len(sample_to_learn['x_to_fit'][0])
    all_indices = list(range(seq_len))
    if seq_len >= params.max_input_points:
        random_indices = random.sample(all_indices, params.max_input_points)
        sub_sample['X_scaled_to_fit'][0] = np.array(
            [sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices]
        )
        sub_sample['Y_scaled_to_fit'][0] = np.array(
            [sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices]
        )
        sub_sample['x_to_fit'][0] = np.array(
            [sample_to_learn['x_to_fit'][0][i] for i in random_indices]
        )
        sub_sample['y_to_fit'][0] = np.array(
            [sample_to_learn['y_to_fit'][0][i] for i in random_indices]
        )

        seq_len = len(sample_to_learn['x_to_predict'][0])
        all_indices = list(range(seq_len))
        if seq_len >= params.max_input_points:
            random_indices = random.sample(all_indices, params.max_input_points)
            sub_sample['x_to_predict'][0] = np.array(
                [sample_to_learn['x_to_predict'][0][i] for i in random_indices]
            )
            sub_sample['y_to_predict'][0] = np.array(
                [sample_to_learn['y_to_predict'][0][i] for i in random_indices]
            )
        else:
            all_indices = list(range(len(sample_to_learn['x_to_fit'][0])))
            random_indices = random.sample(all_indices, params.max_input_points)
            sub_sample['x_to_predict'][0] = np.array(
                [sample_to_learn['x_to_fit'][0][i] for i in random_indices]
            )
            sub_sample['y_to_predict'][0] = np.array(
                [sample_to_learn['y_to_fit'][0][i] for i in random_indices]
            )

    else:
        sub_sample = sample_to_learn

    outputs = model(sub_sample, max_len=params.max_target_len, return_logvar=True)
    encoded_y, generations, gen_len, initial_prior_logvar = outputs
    stored_skeletons = set()

    global_best = {
        'max_fitness': -float('inf'),
        'min_mse': float('inf'),
        'best_eq': 'NaN',
        'best_r2': 0,
        'best_complexity': -1,
    }

    try:
        result = evaluate_candidate(encoded_y, generations,
                                    sample_to_learn, stored_skeletons)
        if result['success']:
            update_global_best(result, global_best)
            if 'skeleton_key' in result:
                stored_skeletons.add(result['skeleton_key'])

            batch_results["direct_predicted_tree"].extend([result['tree']])
            batch_results["complexity_gt_tree"].extend([result['complexity']])
            batch_results["direct_fit_mse"].extend([result['mse_fit']])
            batch_results["direct_pred_mse"].extend([result['mse_pred']])

            for k, v in result['results_fit'].items():
                batch_results[k + "_direct_fit"].extend(v)
            for k, v in result['results_predict'].items():
                batch_results[k + "_direct_predict"].extend(v)
        else:
            batch_results["direct_predicted_tree"].extend(['NaN'])
            batch_results["complexity_gt_tree"].extend([-1])
            batch_results["direct_fit_mse"].extend([0])
            batch_results["direct_pred_mse"].extend([0])

            metrics_keys = [
                'r2_zero', 'r2', 'accuracy_l1_biggio', 'accuracy_l1_1e-3',
                'accuracy_l1_1e-2', 'accuracy_l1_1e-1', '_complexity',
            ]
            for k in metrics_keys:
                batch_results[k + "_direct_fit"].extend([0])
                batch_results[k + "_direct_predict"].extend([0])

    except Exception as e:
        batch_results["direct_predicted_tree"].extend(['NaN'])
        batch_results["complexity_gt_tree"].extend([-1])
        batch_results["direct_fit_mse"].extend([0])
        batch_results["direct_pred_mse"].extend([0])

        metrics_keys = [
            'r2_zero', 'r2', 'accuracy_l1_biggio', 'accuracy_l1_1e-3',
            'accuracy_l1_1e-2', 'accuracy_l1_1e-1', '_complexity',
        ]
        for k in metrics_keys:
            batch_results[k + "_direct_fit"].extend([0])
            batch_results[k + "_direct_predict"].extend([0])

    pop, all_prior_logvars = create_evo_population_covfromvae(
        env, params, model, sample_to_learn, encoded_y, initial_prior_logvar
    )
    lb = torch.min(pop).item()
    ub = torch.max(pop).item()

    if params.use_cma_compre:
        cma = DiagonalCMAES_compre(
            feature_dim=pop.shape[1],
            mu=params.mu_num,
            sigma=params.ev_sigma,
            device='cuda',
            c1_min=params.c1_min,
        )
    else:
        cma = DiagonalCMAES(
            feature_dim=pop.shape[1],
            mu=params.mu_num,
            sigma=params.ev_sigma,
            device='cuda',
            c1_min=params.c1_min,
        )

    max_iteration = params.lso_max_iteration

    start_time_fwd = time.time()
    pop_src_enc = model.prepare_latent_for_decoder(
        pop, initial_prior_logvar.expand(pop.shape[0], -1)
    )
    generations_pop = model.generate_from_latent_sampling(pop_src_enc)

    # ── Set up multiprocessing pool (BFGS is pure numpy, fork-safe) ────
    global _mp_env, _mp_params, _mp_sample
    _mp_env = env
    _mp_params = params
    _mp_sample = sample_to_learn

    num_workers = getattr(params, 'num_eval_workers', 1)
    _pool = None
    if num_workers > 1:
        _pool = mp.get_context('fork').Pool(num_workers)

    fitness_pop, r2_pop, _ = _evaluate_candidates(
        pop, generations_pop, sample_to_learn, set(), num_workers,
    )

    sorted_indices = torch.argsort(fitness_pop, descending=True)
    best_mu_indices = sorted_indices[:params.mu_num]

    best_prior_logvars = [all_prior_logvars[idx.item()] for idx in best_mu_indices]
    best_prior_logvars_stacked = torch.stack(best_prior_logvars, dim=0)

    weights = cma.weights.view(-1, 1, 1)
    weighted_prior_logvar = (best_prior_logvars_stacked * weights).sum(dim=0)
    mean_prior_logvar = weighted_prior_logvar.squeeze(0)

    initial_cov_diag = torch.exp(mean_prior_logvar)
    initial_cov_diag = (
        (initial_cov_diag - initial_cov_diag.min())
        / (initial_cov_diag.max() - initial_cov_diag.min())
    )

    if params.use_cma_compre:
        _, top_indices = torch.topk(initial_cov_diag, params.compre_num)
        mask = torch.zeros_like(initial_cov_diag)
        mask[top_indices] = 1.0
        initial_cov_diag = initial_cov_diag * mask
        cma.mask = mask

    cma.cov_diag = initial_cov_diag

    start_time = time.time()
    iteration_times = []

    mean = cma.tell(pop, fitness_pop, pop)

    for t in range(max_iteration):
        iteration_start_time = time.time()

        candidates = cma.ask(mean, pop_size=pop.shape[0], lb=1.5 * lb, ub=1.5 * ub)

        try:
            cand_src_enc = model.prepare_latent_for_decoder(
                candidates, initial_prior_logvar.expand(candidates.shape[0], -1)
            )
            candidate_generations = model.generate_from_latent_sampling(cand_src_enc)
        except Exception as e:

            metrics_keys = [
                'r2_zero', 'r2', 'accuracy_l1_biggio', 'accuracy_l1_1e-3',
                'accuracy_l1_1e-2', 'accuracy_l1_1e-1', '_complexity',
            ]
            for k in metrics_keys:
                if k + "_final_fit" not in batch_results:
                    batch_results[k + "_final_fit"] = [0.0]
                if k + "_final_predict" not in batch_results:
                    batch_results[k + "_final_predict"] = [0.0]

            optimization_duration = time.time() - start_time
            batch_results["time"].extend([optimization_duration])
            batch_results["all_iteration_times"] = iteration_times
            if _pool is not None:
                _pool.close()
                _pool.join()
            return batch_results

        fitness_values, _, gen_false_num = _evaluate_candidates(
            candidates, candidate_generations, sample_to_learn,
            stored_skeletons, num_workers,
        )

        mean = cma.tell(candidates, fitness_values, candidates)

        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        iteration_times.append(iteration_duration)

        if t < params.warmup_iteration:
            if global_best['best_r2'] > 0.99:
                break
        else:
            if global_best['best_r2'] > params.lso_stop_r2:
                break

        if t > params.lso_min_iteration:
            if t > 30:
                if global_best['best_r2'] < params.lso_stop_r2_abandon_lower:
                    break

    optimization_duration = time.time() - start_time
    batch_results["final_predicted_tree"].extend([global_best['best_eq']])
    batch_results["final_fit_mse"].extend([global_best['min_mse']])
    batch_results["time"].extend([optimization_duration])

    dimension = sample_to_learn['x_to_fit'][0].shape[1]
    x_gt = sample_to_learn['x_to_fit'][0].reshape(-1, dimension)
    y_gt = sample_to_learn['y_to_fit'][0].reshape(-1, 1)
    x_gt_pred = sample_to_learn['x_to_predict'][0].reshape(-1, dimension)
    y_gt_pred = sample_to_learn['y_to_predict'][0].reshape(-1, 1)

    try:
        numexpr_fn = env.simplifier.tree_to_numexpr_fn(global_best['best_eq'])
        y = numexpr_fn(x_gt)[:, 0].reshape(-1, 1)
        y_pred = numexpr_fn(x_gt_pred)[:, 0].reshape(-1, 1)

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(y_pred)):
            mse_fit = float('inf')
            mse_pred = float('inf')
        else:
            y = np.clip(y, -1e150, 1e150)
            y_pred = np.clip(y_pred, -1e150, 1e150)
            mse_fit = np.mean((y - y_gt)**2) / (np.mean(y_gt**2) + 1e-10)
            mse_pred = np.mean((y_pred - y_gt_pred)**2) / (np.mean(y_gt**2) + 1e-10)

        complexity = len(global_best['best_eq'].prefix().split(','))

        results_fit = compute_metrics(
            {
                "true": [y_gt],
                "predicted": [y],
                "predicted_tree": [global_best['best_eq']],
            },
            metrics=params.validation_metrics,
        )

        results_predict = compute_metrics(
            {
                "true": [y_gt_pred],
                "predicted": [y_pred],
                "predicted_tree": [global_best['best_eq']],
            },
            metrics=params.validation_metrics,
        )

        for k, v in results_fit.items():
            batch_results[k + "_final_fit"].extend(v)
        del results_fit

        for k, v in results_predict.items():
            batch_results[k + "_final_predict"].extend(v)
        del results_predict

        batch_results["all_iteration_times"] = iteration_times
        batch_results["es_total_iterations"] = len(iteration_times)
        if iteration_times:
            batch_results["es_avg_iteration_time"] = [np.mean(iteration_times)]
            batch_results["es_max_iteration_time"] = [np.max(iteration_times)]
            batch_results["es_min_iteration_time"] = [np.min(iteration_times)]

    except Exception as e:

        metrics_keys = [
            'r2_zero', 'r2', 'accuracy_l1_biggio', 'accuracy_l1_1e-3',
            'accuracy_l1_1e-2', 'accuracy_l1_1e-1', '_complexity',
        ]
        for k in metrics_keys:
            if k + "_final_fit" not in batch_results:
                batch_results[k + "_final_fit"] = [0.0]
            if k + "_final_predict" not in batch_results:
                batch_results[k + "_final_predict"] = [0.0]

        batch_results["all_iteration_times"] = iteration_times
        batch_results["es_total_iterations"] = len(iteration_times)
        if iteration_times:
            batch_results["es_avg_iteration_time"] = [np.mean(iteration_times)]
            batch_results["es_max_iteration_time"] = [np.max(iteration_times)]
            batch_results["es_min_iteration_time"] = [np.min(iteration_times)]

    if _pool is not None:
        _pool.close()
        _pool.join()

    return batch_results

