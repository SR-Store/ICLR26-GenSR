
import argparse
from symbolicregression.envs import ENVS
from symbolicregression.utils import bool_flag


def get_parser():
    parser = argparse.ArgumentParser(description="Function prediction", add_help=False)

    parser.add_argument(
        "--lso_optimizer", type=str, default="es_fromvae_fit", help="LSO optimizer"
    )

    parser.add_argument(
        "--es_strategy", type=str, default="adaptive_gaussian",
        help="Evolution strategy variant: 'adaptive_gaussian' or 'cma_es'"
    )

    parser.add_argument(
        "--dump_path", type=str, default="", help="Experiment dump path"
    )
    parser.add_argument(
        "--refinements_types",
        type=str,
        default="method=BFGS_batchsize=256_metric=/_mse",
        help="What refinement to use. Separate each arg by _ and value by =. None disables refinement.",
    )

    parser.add_argument(
        "--seed", type=int, default=23, help="Random seed for LSO experiments"
    )

    parser.add_argument(
        "--lso_max_iteration", type=int, default=50, help="Maximum iterations for LSO"
    )

    parser.add_argument(
        "--lso_min_iteration", type=int, default=1, help="Minimum iterations required for LSO"
    )

    parser.add_argument(
        "--warmup_iteration", type=int, default=11, help="Number of warmup iterations with stricter early-stop threshold"
    )

    parser.add_argument(
        "--lso_stop_r2", type=float, default=0.995, help="R2 stop criterion for LSO"
    )

    parser.add_argument(
        "--lso_stop_r2_abandon_lower", type=float, default=0, help="Lower R2 threshold for abandoning LSO"
    )

    parser.add_argument(
        "--num_eval_workers", type=int, default=1, help="Number of threads for parallel candidate evaluation (BFGS is numpy-based, GIL-free)"
    )

    parser.add_argument(
        "--eval_dump_path", type=str, default=None, help="Evaluation dump path"
    )
    parser.add_argument(
        "--save_results", type=bool, default=True, help="Whether to save results"
    )

    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Print every n steps"
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=25,
        help="Save the model periodically (0 to disable)",
    )
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")

    parser.add_argument(
        "--fp16", type=bool_flag, default=False, help="Run model with float16"
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=-1,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )
    parser.add_argument(
        "--rescale", type=bool, default=True, help="Whether to rescale at inference.",
    )

    parser.add_argument(
        "--embedder_type",
        type=str,
        default="LinearPoint",
        help="[TNet, LinearPoint, Flat, AttentionPoint] How to pre-process sequences before passing to a transformer.",
    )

    parser.add_argument(
        "--normalize_y",
        type=bool_flag,
        default=False,
        help="Normalize y for pretraining",
    )

    parser.add_argument(
        "--latent_dim", type=int, default=512, help="Dimension of latent space (bottleneck)"
    )

    parser.add_argument(
        "--emb_emb_dim", type=int, default=64, help="Embedder embedding layer size"
    )
    parser.add_argument(
        "--enc_emb_dim", type=int, default=512, help="Encoder embedding layer size"
    )
    parser.add_argument(
        "--dec_emb_dim", type=int, default=512, help="Decoder embedding layer size"
    )
    parser.add_argument(
        "--n_emb_layers", type=int, default=1, help="Number of layers in the embedder",
    )
    parser.add_argument(
        "--n_enc_layers",
        type=int,
        default=8,
        help="Number of Transformer layers in the encoder",
    )
    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=16,
        help="Number of Transformer layers in the decoder",
    )
    parser.add_argument(
        "--n_enc_heads",
        type=int,
        default=16,
        help="Number of Transformer encoder heads",
    )
    parser.add_argument(
        "--n_dec_heads",
        type=int,
        default=16,
        help="Number of Transformer decoder heads",
    )
    parser.add_argument(
        "--emb_expansion_factor",
        type=int,
        default=1,
        help="Expansion factor for embedder",
    )
    parser.add_argument(
        "--n_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer encoder",
    )
    parser.add_argument(
        "--n_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer decoder",
    )

    parser.add_argument(
        "--norm_attention",
        type=bool_flag,
        default=False,
        help="Normalize attention and train temperature in Transformer",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )
    parser.add_argument(
        "--enc_positional_embeddings",
        type=str,
        default="learnable",
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--dec_positional_embeddings",
        type=str,
        default="learnable",
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )

    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=0,
        help="Base seed for environments (-1 to use timestamp seed)",
    )
    parser.add_argument(
        "--test_env_seed", type=int, default=1, help="Test seed for environments"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Number of sentences per batch"
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=64,
        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_inverse_sqrt,warmup_updates=10000",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.5,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--n_steps_per_epoch", type=int, default=3000, help="Number of steps per epoch",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Number of epochs"
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )

    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of CPU workers for DataLoader",
    )

    parser.add_argument(
        "--train_noise_gamma",
        type=float,
        default=0.0,
        help="Additional output noise during training",
    )

    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="Split into chunks of size max_input_points at eval",
    )
    parser.add_argument(
        "--n_trees_to_refine", type=int, default=10, help="Refine top n trees"
    )

    parser.add_argument(
        "--export_data",
        type=bool_flag,
        default=False,
        help="Export data and disable training.",
    )
    parser.add_argument(
        "--reload_data",
        type=str,
        default="",
        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)",
    )
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )
    parser.add_argument(
        "--batch_load",
        type=bool_flag,
        default=False,
        help="Load training set by batches (of size reload_size).",
    )

    parser.add_argument(
        "--env_name", type=str, default="functions", help="Environment name"
    )

    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    parser.add_argument("--tasks", type=str, default="functions", help="Tasks")

    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=True,
        help="Evaluate with beam search decoding.",
    )
    parser.add_argument(
        "--max_generated_output_len",
        type=int,
        default=200,
        help="Max generated output length",
    )
    parser.add_argument(
        "--beam_eval_train",
        type=int,
        default=0,
        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="Beam size, default = 1 (greedy decoding)",
    )
    parser.add_argument(
        "--beam_type", type=str, default="sampling", help="Beam search or sampling",
    )
    parser.add_argument(
        "--beam_temperature",
        type=int,
        default=0.1,
        help="Beam temperature for sampling",
    )

    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )

    parser.add_argument("--max_number_bags", type=int, default=1)

    parser.add_argument(
        "--reload_model", type=str, default="", help="Reload a pretrained model"
    )

    parser.add_argument(
        "--reload_model_dir", type=str, default="", help="Reload a pretrained model from directory"
    )

    parser.add_argument(
        "--reload_checkpoint", type=str, default="", help="Reload a checkpoint"
    )

    parser.add_argument(
        "--validation_metrics",
        type=str,
        default="r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity",
        help="What metrics should we report? accuracy_tolerance/_l1_error/r2/_complexity/_relative_complexity/is_symbolic_solution",
    )

    parser.add_argument(
        "--debug_train_statistics",
        type=bool,
        default=False,
        help="Whether to print distribution info",
    )

    parser.add_argument(
        "--eval_noise_gamma",
        type=float,
        default=0.0,
        help="Additional output noise during evaluation",
    )
    parser.add_argument(
        "--eval_size", type=int, default=10000, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_noise_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Type of noise added at test time",
    )
    parser.add_argument(
        "--eval_noise", type=float, default=0, help="Noise level for evaluation"
    )
    parser.add_argument(
        "--eval_only", type=bool_flag, default=False, help="Only run evaluations"
    )
    parser.add_argument(
        "--eval_from_exp", type=str, default="", help="Path of experiment to use"
    )
    parser.add_argument(
        "--eval_data", type=str, default="", help="Path of data to eval"
    )
    parser.add_argument(
        "--eval_verbose", type=int, default=0, help="Export evaluation details"
    )
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )
    parser.add_argument(
        "--eval_input_length_modulo",
        type=int,
        default=-1,
        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation",
    )
    parser.add_argument("--eval_on_pmlb", type=bool, default=False)
    parser.add_argument("--eval_in_domain", type=bool, default=True)
    parser.add_argument("--eval_lso_on_pmlb", type=bool, default=False)
    parser.add_argument("--eval_lso_in_domain", type=bool, default=True)
    parser.add_argument(
        "--pmlb_data_type",
        type=str,
        default="feynman",
        help="PMLB dataset type",
    )
    parser.add_argument(
        "--target_noise",
        type=float,
        default=0.0,
        help="Target noise added to y_to_fit for PMLB evaluation",
    )

    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--windows",
        type=bool_flag,
        default=False,
        help="Windows version (no multiprocessing for eval)",
    )
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )

    parser.add_argument(
        "--max_src_len", type=int, default=200, help="Max source length")
    parser.add_argument(
        "--max_target_len", type=int, default=200, help="Max target length")
    parser.add_argument(
        "--run", type=int, default=None, help="Run ID")

    parser.add_argument(
        "--filter_feynman_subset", type=bool, default=False, help="Filter Feynman problems to a predefined subset"
    )

    parser.add_argument(
        "--run_name", type=str, default="debug", help="Run name for logging")

    parser.add_argument(
        "--ev_sigma", type=float, default=0.01, help="Evolution strategy sigma"
    )

    parser.add_argument(
        "--problem_name", type=str, default="feynman_III_9_52", help="Problem name"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="vae",
        help="Type of model to use: 'vae' for CVAE-based symbolic regression"
    )

    parser.add_argument(
        "--d_model", type=int, default=512, help="Model dimension for transformer layers"
    )
    parser.add_argument(
        "--d_input", type=int, default=512, help="Input embedding dimension"
    )
    parser.add_argument(
        "--n_TE_layers", type=int, default=8, help="Number of transformer encoder layers in CVAE model"
    )
    parser.add_argument(
        "--kl_annealing_steps", type=int, default=None, help="Number of steps for KL annealing in VAE (None for no annealing)"
    )
    parser.add_argument(
        "--uniform_sample_number", type=int, default=100, help="Number of uniform sample points for numerical data"
    )

    parser.add_argument(
        "--feynman_sel_equs_num", type=int, default=2, help="Number of Feynman equations to select"
    )

    parser.add_argument(
        "--pop_num", type=int, default=15, help="Population size for the first generation"
    )

    parser.add_argument(
        "--mu_num", type=int, default=2, help="Number of parents"
    )

    parser.add_argument(
        "--wandb_disabled", action="store_true", help="Disable wandb"
    )

    parser.add_argument(
        "--kl_limits", type=float, default=0.2, help="KL divergence limits"
    )

    parser.add_argument(
        "--eval_in_train_mode", action="store_true", help="Evaluate in train mode"
    )

    parser.add_argument(
        "--test_flag", action="store_true", help="Test flag"
    )

    parser.add_argument(
        "--c1_min", type=float, default=-1, help="Minimum c1 value for CMA-ES"
    )

    parser.add_argument('--pop_init_index', type=str, default="1*0*0", help='Control use_sample_pop, use_y_noise_pop, use_latent_noise_pop')

    parser.add_argument(
        "--use_sample_pop", type=int, default=1, help="Use sample-based population initialization"
    )

    parser.add_argument(
        "--use_y_noise_pop", type=int, default=0, help="Use y-noise-based population initialization"
    )

    parser.add_argument(
        "--use_latent_noise_pop", type=int, default=0, help="Use latent-noise-based population initialization"
    )

    parser.add_argument(
        "--use_cma_compre", type=int, default=0, help="Use CMA with compression"
    )

    parser.add_argument(
        "--compre_num", type=int, default=128, help="Dimension after compression"
    )

    parser.add_argument(
        "--monitor_iter", type=int, default=100, help="Monitor iteration interval"
    )

    parser.add_argument(
        "--wandb_group_name", type=str, default="test", help="Wandb group name"
    )

    parser.add_argument(
        "--wandb_run_name", type=str, default="", help="Wandb run name (auto-generated if empty)"
    )

    parser.add_argument(
        "--wandb_project", type=str, default="symbolic-regression-training", help="Wandb project"
    )

    parser.add_argument(
        "--sweep_problems", type=int, default=0, help="Sweep problems"
    )

    parser.add_argument(
        "--is_epoch_evaluation", type=int, default=1, help="Enable epoch-level evaluation"
    )

    parser.add_argument(
        "--com_weight", type=float, default=100.0, help="Complexity weight"
    )

    parser.add_argument(
        "--reverse_hard_probelms", type=int, default=-1, help="Reverse hard problems"
    )

    parser.add_argument(
        "--wandb_resume_id", type=str, default="", help="Resume a wandb run by ID (uses resume='must')"
    )

    parser.add_argument(
        "--all_half_thre", type=float, default=0.983, help="Threshold for all-half evaluation"
    )

    return parser

