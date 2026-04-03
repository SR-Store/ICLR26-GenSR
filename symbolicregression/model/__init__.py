
from logging import getLogger
import os
import torch
from .embedders import NumericalEmbedder
from .transformer import TransformerModel
from .cvae import CVAEDE_SR
from .feature_fusion import FeatureFusion

from .sklearn_wrapper import SymbolicTransformerRegressor
from .model_wrapper import ModelWrapper
import torch.nn as nn

logger = getLogger()


def reload_model(modules, modules_to_load, path=None, requires_grad=True):
    model_path = path
    assert os.path.isfile(model_path)

    logger.warning(f"Reloading pretrained model from {model_path} ...")

    if not torch.cuda.is_available():
        data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        data = torch.load(model_path, weights_only=False)

    for k in modules_to_load:
        v = modules[k]
        weights = data[k]
        try:
            weights = data[k]
            v.load_state_dict(weights)
            print(f"Loaded {k} parameters from {model_path}")
        except RuntimeError:
            weights = {name.partition(".")[2]: v for name, v in data[k].items()}
            v.load_state_dict(weights)

        v.requires_grad = requires_grad

    return modules


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def check_model_params(params):
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def count_module_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def analyze_cvaede_sr_parameters(model, model_name="CVAEDE_SR", model_type="vae_v1"):
    if not isinstance(model, CVAEDE_SR):
        logger.info(f"Skipping detailed analysis for {model_name} (not CVAEDE_SR)")
        return

    logger.info("=" * 80)
    logger.info(f"DETAILED PARAMETER ANALYSIS FOR {model_name}")
    logger.info("=" * 80)

    total_params = count_module_parameters(model) / 1e6
    logger.info(f"Total Parameters: {total_params:.3f}M")
    logger.info("=" * 80)

    logger.info("CORE COMPONENTS:")
    core_components = [
        ('Token Embeddings', model.token_embeddings),
        ('Numerical Compressor', model.numerical_compressor),
        ('Equation Upper Layer', model.eq_upper),
        ('Scope Embeddings', model.scope_embeddings),
        ('Positional Encoding', model.pos_encoding),
        ('Input Layer', model.input_layer),
    ]

    for name, module in core_components:
        params = count_module_parameters(module) / 1e6
        percentage = (params / total_params) * 100
        logger.info(f"  {name:<25}: {params:>8.3f}M ({percentage:>5.1f}%)")

    query_params = model.query.numel() / 1e6
    query_percentage = (query_params / total_params) * 100
    logger.info(f"  {'Query Parameter':<25}: {query_params:>8.3f}M ({query_percentage:>5.1f}%)")

    logger.info("ENCODER LAYERS:")
    total_encoder_params = 0
    encoder_layer_params = 0

    if hasattr(model, 'encoder_layers') and len(model.encoder_layers) > 0:
        first_layer = model.encoder_layers[0]
        components = [
            'self_attention', 'norm1', 'feedforward', 'norm2',
            'post_attention', 'post_norm1', 'post_feedforward', 'post_norm2'
        ]

        layer_total = 0
        logger.info(f"  Per Layer Components (Layer 0):")
        for comp in components:
            if comp in first_layer:
                comp_params = count_module_parameters(first_layer[comp]) / 1e6
                layer_total += comp_params * 1e6
                logger.info(f"    {comp:<20}: {comp_params:>6.3f}M")

        encoder_layer_params = layer_total / 1e6
        total_encoder_params = encoder_layer_params * len(model.encoder_layers)

        logger.info(f"  Parameters per layer     : {encoder_layer_params:>8.3f}M")
        logger.info(f"  Number of layers         : {len(model.encoder_layers)}")
        logger.info(f"  Total encoder parameters : {total_encoder_params:>8.3f}M ({(total_encoder_params/total_params)*100:>5.1f}%)")

    logger.info("ATTENTION & LATENT COMPONENTS:")
    attention_components = [
        ('Concentrate Attention', model.concentrate_attention),
        ('Posterior FC', model.post_fc),
        ('Prior FC1', model.prior_fc1),
        ('Prior FC2', model.prior_fc2),
    ]

    for name, module in attention_components:
        params = count_module_parameters(module) / 1e6
        percentage = (params / total_params) * 100
        logger.info(f"  {name:<25}: {params:>8.3f}M ({percentage:>5.1f}%)")

    logger.info("OUTPUT LAYERS:")
    output_components = [
        ('Last Layer', model.last_layer),
        ('Output Layer', model.output_layer),
    ]

    for name, module in output_components:
        params = count_module_parameters(module) / 1e6
        percentage = (params / total_params) * 100
        logger.info(f"  {name:<25}: {params:>8.3f}M ({percentage:>5.1f}%)")

    logger.info("SUMMARY BY CATEGORY:")

    embedding_params = (count_module_parameters(model.token_embeddings) +
                       count_module_parameters(model.scope_embeddings)) / 1e6

    processing_params = (count_module_parameters(model.numerical_compressor) +
                        count_module_parameters(model.eq_upper) +
                        count_module_parameters(model.input_layer) +
                        model.query.numel()) / 1e6

    encoder_attention_params = (total_encoder_params +
                               count_module_parameters(model.concentrate_attention) / 1e6)

    latent_params = (count_module_parameters(model.post_fc) +
                    count_module_parameters(model.prior_fc1) +
                    count_module_parameters(model.prior_fc2)) / 1e6

    output_params = (count_module_parameters(model.last_layer) +
                    count_module_parameters(model.output_layer)) / 1e6

    categories = [
        ('Embedding Layers', embedding_params),
        ('Processing Layers', processing_params),
        ('Encoder + Attention', encoder_attention_params),
        ('Latent Space', latent_params),
        ('Output Layers', output_params)
    ]

    for name, params in categories:
        percentage = (params / total_params) * 100
        logger.info(f"  {name:<25}: {params:>8.3f}M ({percentage:>5.1f}%)")

    logger.info("KEY INSIGHTS:")
    largest_category = max(categories, key=lambda x: x[1])
    logger.info(f"  Largest component: {largest_category[0]} ({largest_category[1]:.1f}M)")
    if hasattr(model, 'encoder_layers') and len(model.encoder_layers) > 0:
        logger.info(f"  Parameters per encoder layer: {encoder_layer_params:.1f}M")
    if hasattr(model, 'token_list'):
        efficiency = total_params / len(model.token_list)
        logger.info(f"  Model efficiency: {efficiency:.3f}M params per token")

    logger.info("=" * 80)


def build_modules(env, params, mode='train'):
    modules = {}

    modules['data_encoder'] = NumericalEmbedder(params, env)
    env.get_length_after_batching = modules['data_encoder'].get_length_after_batching

    modules["cvae"] = CVAEDE_SR(params, env)

    modules["token_embed"] = Embedding(
        len(env.equation_words), params.d_input, padding_idx=82
    )

    modules["seq_decoder"] = TransformerModel(
        params,
        env.equation_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
        rebuttal=1
    )

    modules["feature_fusion"] = FeatureFusion(
        latent_dim=params.latent_dim,
        dec_emb_dim=params.dec_emb_dim,
        seq_len=params.max_src_len,
    )

    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum(p.numel() for p in v.parameters() if p.requires_grad) / 1_000_000:.4f}M"
        )

    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules

