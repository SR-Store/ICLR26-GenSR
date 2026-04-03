from torch import nn
import torch

__all__ = ['VAESymbolicRegressor']


class VAESymbolicRegressor(nn.Module):
    def __init__(self, params, env, modules):
        super().__init__()
        self.modules = modules
        self.params = params
        self.env = env

        self.vae_model = self.modules["cvae"]
        self.decoder = self.modules["seq_decoder"]
        self.embedder_f = self.modules["data_encoder"]
        self.embedder_e = self.modules["token_embed"]
        self.feature_fusion = self.modules["feature_fusion"]

        self.vae_model.eval()
        self.decoder.eval()
        self.embedder_f.eval()
        self.embedder_e.eval()
        self.feature_fusion.eval()

    def prepare_latent_for_decoder(self, z, logvar):
        return self.feature_fusion(z, logvar)

    def generate_from_latent(self, src_enc):
        generations, gen_len = self.decoder.generate_from_latent(
            src_enc,
            sample_temperature=None,
            max_len=self.params.max_target_len,
        )
        generations = generations.transpose(0, 1)
        return generations, gen_len

    def generate_from_latent_direct(self, src_enc):
        bs = src_enc.shape[0]
        num_samples = 10
        encoded = (
            src_enc.unsqueeze(1)
            .expand((bs, num_samples) + src_enc.shape[1:])
            .contiguous()
            .view((bs * num_samples,) + src_enc.shape[1:])
        )
        sampling_generations, gen_len = self.decoder.generate_from_latent(
            encoded,
            sample_temperature=self.params.beam_temperature,
            max_len=self.params.max_generated_output_len,
        )
        generations = sampling_generations.transpose(0, 1)
        return generations, gen_len

    def generate_from_latent_sampling(self, src_enc):
        bs = src_enc.shape[0]
        num_samples = self.params.beam_size
        encoded = (
            src_enc.unsqueeze(1)
            .expand((bs, num_samples) + src_enc.shape[1:])
            .contiguous()
            .view((bs * num_samples,) + src_enc.shape[1:])
        )
        sampling_generations, _ = self.decoder.generate_from_latent(
            encoded,
            sample_temperature=self.params.beam_temperature,
            max_len=self.params.max_generated_output_len,
        )
        generations = sampling_generations.transpose(0, 1)
        return generations

    def _encode(self, samples):
        x1 = []
        for seq_id in range(len(samples["X_scaled_to_fit"])):
            x1.append([])
            for seq_l in range(len(samples["X_scaled_to_fit"][seq_id])):
                x1[seq_id].append([
                    samples["X_scaled_to_fit"][seq_id][seq_l],
                    samples["Y_scaled_to_fit"][seq_id][seq_l],
                ])

        x1, len1 = self.embedder_f(x1)

        batch_size = x1.shape[1]
        dummy_eq_tokens = torch.zeros(200, batch_size, 512, device=x1.device)
        dummy_len = len1

        prior_mu, prior_logvar, _, _, _, _, _ = (
            self.vae_model(x1, dummy_eq_tokens, len1, dummy_len, mode="infer")
        )
        return prior_mu, prior_logvar

    def encode_only(self, samples):
        prior_mu, prior_logvar = self._encode(samples)
        return prior_mu, prior_logvar

    def forward(self, samples, max_len, return_logvar=False):
        prior_mu, prior_logvar = self._encode(samples)

        latent_repr = self.feature_fusion(prior_mu, prior_logvar)

        generations, gen_len = self.generate_from_latent_direct(latent_repr)

        outputs = (prior_mu, generations, gen_len)
        if return_logvar:
            outputs = (prior_mu, generations, gen_len, prior_logvar)
        return outputs

