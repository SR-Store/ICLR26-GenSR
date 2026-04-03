
from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_MAX_POSITIONS = 4096


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal):
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class ScaledDotAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.layer_id = next(ScaledDotAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(src_dim, dim)
        self.value_proj = nn.Linear(src_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )

    def forward(self, input, mask=None, kv=None, use_cache=False):
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def shape(x):
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.query_proj(input))
        if kv is None:
            k = shape(self.key_proj(input))
            v = shape(self.value_proj(input))
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.key_proj(k))
            v = shape(self.value_proj(v))

        if use_cache:
            if self.layer_id in self.cache:
                if kv is None:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)

        scores = torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
            mask = (
                (mask == 0).view(mask_reshape).expand_as(scores)
            )
            scores.masked_fill_(mask, -float("inf"))

        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )
        weights = F.dropout(
            weights, p=self.dropout, training=self.training
        )
        context = torch.matmul(weights, v)
        context = unshape(context)

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = weights.detach().cpu()

        return self.out_proj(context)


class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.mid_layers = nn.ModuleList()
        self.fc_in = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.mid_layers.append(nn.Linear(dim_hidden, dim_hidden))
        self.fc_out = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.fc_in(input)
        x = F.relu(x)
        for mlin in self.mid_layers:
            x = mlin(x)
            x = F.relu(x)
        x = self.fc_out(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PoolingHead(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(PoolingHead, self).__init__()

        self.compress = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h):

        z_rep = self.compress(h)

        return z_rep


class CVAETransformer(nn.Module):

    STORE_OUTPUTS = True

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
    ):
        super().__init__()

        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        self.apex = params.nvidia_apex


        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]

        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words

        self.latent_dim = params.latent_dim
        self.dim = (
            params.enc_emb_dim if is_encoder else params.dec_emb_dim
        )
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = (
            params.n_enc_heads if is_encoder else params.n_dec_heads
        )
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"


        if positional_embeddings is None or positional_embeddings == "alibi":
            self.pos_embed = None
        elif positional_embeddings == "sinusoidal":
            self.pos_embed = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.pos_embed.weight
            )
        elif positional_embeddings == "learnable":
            self.pos_embed = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.tok_embed = Embedding(
                self.n_words, self.dim, padding_idx=self.pad_index
            )
        else:
            self.tok_embed = None
        self.input_norm = nn.LayerNorm(self.dim, eps=1e-12)

        self.self_attn = nn.ModuleList()
        self.attn_norm = nn.ModuleList()
        self.feed_forward = nn.ModuleList()
        self.ff_norm = nn.ModuleList()
        if self.is_decoder:
            self.cross_norm = nn.ModuleList()
            self.cross_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.self_attn.append(
                ScaledDotAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.attn_norm.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.cross_norm.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.cross_attn.append(
                    ScaledDotAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.feed_forward.append(
                FeedForwardBlock(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.ff_norm.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

        if self.with_output:
            assert not self.use_prior_embeddings
            self.lm_head = nn.Linear(
                self.dim, self.n_words, bias=True
            )
            if params.share_inout_emb:
                self.lm_head.weight = self.tok_embed.weight

        self.pool_attn = nn.Sequential(
            nn.Linear(self.dim, 1), nn.Softmax(dim=1)
        )

        self.z_rep = None

        self.bottleneck = PoolingHead(self.dim, self.latent_dim)

        self.latent_proj = nn.Linear(self.latent_dim, self.dim)


    def forward(self, mode, **kwargs):
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)

        mask, attn_mask = get_masks(slen, lengths, causal)
        src_mask = None 

        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        if use_cache:
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        if not self.use_prior_embeddings:
            tensor = self.tok_embed(x)
        else:
            tensor = x

        if self.pos_embed is not None:
            tensor = tensor + self.pos_embed(positions).expand_as(tensor)
        tensor = self.input_norm(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        for i in range(self.n_layers):

            self.self_attn[i].cache = self.cache
            attn = self.self_attn[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.attn_norm[i](tensor)

            if self.is_decoder and src_enc is not None:
                self.cross_attn[i].cache = self.cache
                src_enc_proj = self.latent_proj(src_enc)
                src_enc_expanded = src_enc_proj.unsqueeze(1).expand(tensor.shape)
                attn = self.cross_attn[i](
                    tensor, src_mask, kv=src_enc_expanded, use_cache=use_cache
                )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.cross_norm[i](tensor)

            tensor = tensor + self.feed_forward[i](tensor)
            tensor = self.ff_norm[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        if use_cache:
            self.cache["slen"] += tensor.size(1)

        if self.is_encoder:
            glob_attn = self.pool_attn(tensor)
            z_rep = torch.bmm(glob_attn.transpose(-1, 1), tensor).squeeze()

            if len(tensor) == 1:
                z_rep = z_rep.unsqueeze(0)

            z_rep = self.bottleneck(z_rep)
            return z_rep

        if self.is_decoder:
            tensor = tensor.transpose(0, 1)
            return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.lm_head(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def generate(
        self, src_enc, src_len, max_len=200, top_p=1.0, sample_temperature=None
    ):

        bs = len(src_len)
        assert src_enc.size(0) == bs

        generated = src_len.new(max_len, bs)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        self.cache = {"slen": 0}
        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,)
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)
            scores = self.lm_head(tensor)

            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
  
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated, gen_len

    def generate_from_latent(
        self, src_enc, max_len=200, top_p=1.0, sample_temperature=None
    ):

        bs = src_enc.size(0)

        generated = src_enc.new(max_len, bs).long()
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        positions = src_enc.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        cur_len = 1
        gen_len = src_enc.new(bs).long().fill_(1)
        unfinished_sents = src_enc.new(bs).long().fill_(1)

        self.cache = {"slen": 0}
        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=None,
                use_cache=True,)
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)
            scores = self.lm_head(tensor)

            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
  
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated, gen_len

    def generate_beam(
        self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200,
    ):

        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words

        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        generated = src_len.new(max_len, bs * beam_size)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        )

        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        cur_len = 1

        self.cache = {"slen": 0}

        done = [False for _ in range(bs)]

        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            assert tensor.size() == (1, bs * beam_size, self.dim)
            if self.apex:
                tensor = tensor.data[-1, :, :].to(self.dtype)
            else:
                tensor = tensor.data[
                    -1, :, :
                ]
            scores = self.lm_head(tensor)
            scores = F.log_softmax(scores.float(), dim=-1)
            assert scores.size() == (bs * beam_size, n_words)

            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )
            _scores = _scores.view(bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            next_batch_beam = []

            for sent_id in range(bs):

                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )
                    continue

                next_sent_beam = []

                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    beam_id = torch.div(idx, n_words, rounding_mode="trunc")
                    word_id = idx % n_words

                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    if len(next_sent_beam) == beam_size:
                        break

                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0)
                    ] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            cur_len = cur_len + 1

            if all(done):
                break


        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)

        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps


class EncoderTransformer(nn.Module):

    STORE_OUTPUTS = True

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
    ):
        super().__init__()

        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        self.apex = params.nvidia_apex


        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]

        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words

        self.latent_dim = params.latent_dim
        self.dim = (
            params.enc_emb_dim if is_encoder else params.dec_emb_dim
        )
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = (
            params.n_enc_heads if is_encoder else params.n_dec_heads
        )
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"


        if positional_embeddings is None or positional_embeddings == "alibi":
            self.pos_embed = None
        elif positional_embeddings == "sinusoidal":
            self.pos_embed = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.pos_embed.weight
            )
        elif positional_embeddings == "learnable":
            self.pos_embed = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.tok_embed = Embedding(
                self.n_words, self.dim, padding_idx=self.pad_index
            )
        else:
            self.tok_embed = None
        self.input_norm = nn.LayerNorm(self.dim, eps=1e-12)

        self.self_attn = nn.ModuleList()
        self.attn_norm = nn.ModuleList()
        self.feed_forward = nn.ModuleList()
        self.ff_norm = nn.ModuleList()
        if self.is_decoder:
            self.cross_norm = nn.ModuleList()
            self.cross_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.self_attn.append(
                ScaledDotAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.attn_norm.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.cross_norm.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.cross_attn.append(
                    ScaledDotAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.feed_forward.append(
                FeedForwardBlock(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.ff_norm.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

        if self.with_output:
            assert not self.use_prior_embeddings
            self.lm_head = nn.Linear(
                self.dim, self.n_words, bias=True
            )
            if params.share_inout_emb:
                self.lm_head.weight = self.tok_embed.weight

        self.pool_attn = nn.Sequential(
            nn.Linear(self.dim, 1), nn.Softmax(dim=1)
        )

        self.z_rep = None

        self.bottleneck = PoolingHead(self.dim, self.latent_dim)

        self.latent_proj = nn.Linear(self.latent_dim, self.dim)


    def forward(self, mode, **kwargs):
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):

        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)

        mask, attn_mask = get_masks(slen, lengths, causal)
        src_mask = None 

        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        if use_cache:
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        if not self.use_prior_embeddings:
            tensor = self.tok_embed(x)
        else:
            tensor = x

        if self.pos_embed is not None:
            tensor = tensor + self.pos_embed(positions).expand_as(tensor)
        tensor = self.input_norm(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        for i in range(self.n_layers):

            self.self_attn[i].cache = self.cache
            attn = self.self_attn[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.attn_norm[i](tensor)

            if self.is_decoder and src_enc is not None:
                self.cross_attn[i].cache = self.cache
                src_enc_proj = self.latent_proj(src_enc)
                src_enc_expanded = src_enc_proj.unsqueeze(1).expand(tensor.shape)
                attn = self.cross_attn[i](
                    tensor, src_mask, kv=src_enc_expanded, use_cache=use_cache
                )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.cross_norm[i](tensor)

            tensor = tensor + self.feed_forward[i](tensor)
            tensor = self.ff_norm[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        if use_cache:
            self.cache["slen"] += tensor.size(1)

        if self.is_encoder:
            glob_attn = self.pool_attn(tensor)
            z_rep = torch.bmm(glob_attn.transpose(-1, 1), tensor).squeeze()

            if len(tensor) == 1:
                z_rep = z_rep.unsqueeze(0)

            z_rep = self.bottleneck(z_rep)
            return z_rep

        if self.is_decoder:
            tensor = tensor.transpose(0, 1)
            return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.lm_head(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def generate(
        self, src_enc, src_len, max_len=200, top_p=1.0, sample_temperature=None
    ):

        bs = len(src_len)
        assert src_enc.size(0) == bs

        generated = src_len.new(max_len, bs)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        self.cache = {"slen": 0}
        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,)
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)
            scores = self.lm_head(tensor)

            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
  
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated, gen_len

    def generate_from_latent(
        self, src_enc, max_len=200, top_p=1.0, sample_temperature=None
    ):

        bs = src_enc.size(0)

        generated = src_enc.new(max_len, bs).long()
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        positions = src_enc.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        cur_len = 1
        gen_len = src_enc.new(bs).long().fill_(1)
        unfinished_sents = src_enc.new(bs).long().fill_(1)

        self.cache = {"slen": 0}
        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=None,
                use_cache=True,)
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)
            scores = self.lm_head(tensor)

            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
  
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated, gen_len

    def generate_beam(
        self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200,
    ):

        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words

        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        generated = src_len.new(max_len, bs * beam_size)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        )

        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        cur_len = 1

        self.cache = {"slen": 0}

        done = [False for _ in range(bs)]

        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            assert tensor.size() == (1, bs * beam_size, self.dim)
            if self.apex:
                tensor = tensor.data[-1, :, :].to(self.dtype)
            else:
                tensor = tensor.data[
                    -1, :, :
                ]
            scores = self.lm_head(tensor)
            scores = F.log_softmax(scores.float(), dim=-1)
            assert scores.size() == (bs * beam_size, n_words)

            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )
            _scores = _scores.view(bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            next_batch_beam = []

            for sent_id in range(bs):

                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )
                    continue

                next_sent_beam = []

                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    beam_id = torch.div(idx, n_words, rounding_mode="trunc")
                    word_id = idx % n_words

                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    if len(next_sent_beam) == beam_size:
                        break

                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0)
                    ] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            cur_len = cur_len + 1

            if all(done):
                break


        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)

        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps


class TransformerModel(nn.Module): 
    STORE_OUTPUTS = True

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
        rebuttal=None,
    ):
        super().__init__()

        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        self.apex = params.nvidia_apex


        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]

        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words

        self.dim = (
            params.enc_emb_dim if is_encoder else params.dec_emb_dim
        )
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = (
            params.n_enc_heads if is_encoder else params.n_dec_heads
        )
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"


        if positional_embeddings is None or positional_embeddings == "alibi":
            self.pos_embed = None
        elif positional_embeddings == "sinusoidal":
            self.pos_embed = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.pos_embed.weight
            )
        elif positional_embeddings == "learnable":
            self.pos_embed = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.tok_embed = Embedding(
                self.n_words, self.dim, padding_idx=self.pad_index
            )
        else:
            self.tok_embed = None
        self.input_norm = nn.LayerNorm(self.dim, eps=1e-12)

        self.self_attn = nn.ModuleList()
        self.attn_norm = nn.ModuleList()
        self.feed_forward = nn.ModuleList()
        self.ff_norm = nn.ModuleList()
        if self.is_decoder:
            self.cross_norm = nn.ModuleList()
            self.cross_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.self_attn.append(
                ScaledDotAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.attn_norm.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.cross_norm.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.cross_attn.append(
                    ScaledDotAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.feed_forward.append(
                FeedForwardBlock(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.ff_norm.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

        if self.with_output:
            assert not self.use_prior_embeddings
            self.lm_head = nn.Linear(
                self.dim, self.n_words, bias=True
            )
            if params.share_inout_emb:
                self.lm_head.weight = self.tok_embed.weight

    def forward(self, mode, **kwargs):
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)

        mask, attn_mask = get_masks(slen, lengths, causal)
        src_mask = None

        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        if use_cache:
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        if not self.use_prior_embeddings:
            tensor = self.tok_embed(x)
        else:
            tensor = x

        if self.pos_embed is not None:
            tensor = tensor + self.pos_embed(positions).expand_as(tensor)
        tensor = self.input_norm(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        for i in range(self.n_layers):

            self.self_attn[i].cache = self.cache
            attn = self.self_attn[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.attn_norm[i](tensor)

            if self.is_decoder and src_enc is not None:
                self.cross_attn[i].cache = self.cache
                attn = self.cross_attn[i](
                    tensor, src_mask, kv=src_enc, use_cache=use_cache
                )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.cross_norm[i](tensor)

            tensor = tensor + self.feed_forward[i](tensor)
            tensor = self.ff_norm[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        if use_cache:
            self.cache["slen"] += tensor.size(1)

        tensor = tensor.transpose(0, 1)

        return tensor


    def predict(self, tensor, pred_mask, y, get_scores):
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.lm_head(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss


    def generate(
        self, src_enc, src_len, max_len=200, top_p=1.0, sample_temperature=None
    ):

        bs = len(src_len)
        assert src_enc.size(0) == bs

        generated = src_len.new(max_len, bs)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        self.cache = {"slen": 0}
        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,)
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)
            scores = self.lm_head(tensor)

            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
  
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated, gen_len
    
    def generate_from_latent(
        self, src_enc, max_len=200, top_p=1.0, sample_temperature=None
    ):

        bs = src_enc.size(0)

        generated = src_enc.new(max_len, bs).long()
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        positions = src_enc.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        cur_len = 1
        gen_len = src_enc.new(bs).long().fill_(1)
        unfinished_sents = src_enc.new(bs).long().fill_(1)

        self.cache = {"slen": 0}
        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=None,
                use_cache=True,)
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)
            scores = self.lm_head(tensor)

            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
  
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated, gen_len

    def generate_from_latent_with_scores(
        self, src_enc, max_len=200, top_p=1.0, sample_temperature=None
    ):
        bs = src_enc.size(0)

        generated_tokens = [src_enc.new(bs).fill_(self.eos_index)]

        positions = (
            torch.arange(max_len, device=src_enc.device)
            .long()
            .unsqueeze(1)
            .expand(max_len, bs)
        )

        cur_len = 1
        gen_len = src_enc.new(bs).fill_(1)
        unfinished_sents = src_enc.new(bs).fill_(1)

        all_scores = []
        self.cache = {"slen": 0}

        while cur_len < max_len:
            prev_tokens = torch.stack(generated_tokens, dim=0).long()

            tensor = self.forward(
                "fwd",
                x=prev_tokens,
                lengths=gen_len.long(),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=None,
                use_cache=True,
            )
            assert tensor.size() == (1, bs, self.dim)
            scores = self.lm_head(tensor[-1, :, :])
            all_scores.append(scores)

            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1),
                    num_samples=1,
                ).squeeze(1)

            generated_tokens.append(
                next_words * unfinished_sents
                + self.pad_index * (1 - unfinished_sents)
            )

            gen_len = gen_len + unfinished_sents
            unfinished_sents = unfinished_sents * next_words.ne(self.eos_index).long()
            cur_len = cur_len + 1

            if unfinished_sents.max() == 0:
                break


        if cur_len == max_len:
            last_step = (
                generated_tokens[-1] * (1 - unfinished_sents)
                + self.eos_index * unfinished_sents
            )
            generated_tokens[-1] = last_step

        generated = torch.stack(generated_tokens, dim=0)

        all_scores_tensor = torch.stack(all_scores, dim=0)
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated, gen_len, all_scores_tensor


    def extract_top_k(
        self, src_enc, src_len, state, top_k, max_len=200,
    ):
        bs = len(src_len)
        assert src_enc.size(0) == bs

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        state = state.transpose(0,1)


        gen_len = src_len.clone().fill_(len(state))
        self.cache = {"slen": 0}

        tensor = self.forward(
            "fwd",
            x=state,
            lengths=gen_len,
            positions=positions[:len(state)],
            causal=True,
            src_enc=src_enc,
            src_len=src_len,
            use_cache=True,)

        tensor = tensor.data[-1, :, :].to(self.dtype)
        scores = self.lm_head(tensor)
            
        top_k_tokens = torch.topk(scores, top_k)[1].squeeze(1)
      
        return top_k_tokens

    def generate_beam(
        self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200,
    ):

        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words

        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        generated = src_len.new(max_len, bs * beam_size)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)

        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        )

        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        cur_len = 1

        self.cache = {"slen": 0}

        done = [False for _ in range(bs)]

        while cur_len < max_len:

            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            assert tensor.size() == (1, bs * beam_size, self.dim)
            if self.apex:
                tensor = tensor.data[-1, :, :].to(self.dtype)
            else:
                tensor = tensor.data[
                    -1, :, :
                ]
            scores = self.lm_head(tensor)
            scores = F.log_softmax(scores.float(), dim=-1)
            assert scores.size() == (bs * beam_size, n_words)

            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )
            _scores = _scores.view(bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            next_batch_beam = []

            for sent_id in range(bs):

                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )
                    continue

                next_sent_beam = []

                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    beam_id = torch.div(idx, n_words, rounding_mode="trunc")
                    word_id = idx % n_words

                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    if len(next_sent_beam) == beam_size:
                        break

                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0)
                    ] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            cur_len = cur_len + 1

            if all(done):
                break

        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)

        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index
        
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps


    def generate_beam_from_state(
        self, src_enc, src_len, state, beam_size, top_k, top_k_hash, use_prefix_cache, length_penalty, early_stopping, max_len=200,
    ):

        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words

        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        state = state.transpose(0,1)

        generated = src_len.new(max_len, bs * beam_size)
        generated.fill_(self.pad_index)
        generated[:len(state)] = state

        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        )

        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        cur_len = len(state)

        self.cache = {"slen": 0}

        done = [False for _ in range(bs)]
        while cur_len < max_len:
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            if self.apex:
                tensor = tensor.data[-1, :, :].to(self.dtype)
            else:
                tensor = tensor.data[
                    -1, :, :
                ]
            scores = self.lm_head(tensor)
            scores = F.log_softmax(scores.float(), dim=-1)
            assert scores.size() == (bs * beam_size, n_words)

            if use_prefix_cache:
                top_k_tokens = torch.topk(scores, top_k, axis=-1)[1]
                for beam_id in range(len(top_k_tokens)):
                    state_beam = tuple(generated[:cur_len][:,beam_id].tolist())
                    if state_beam not in top_k_hash.keys():
                        top_k_hash[state_beam] = top_k_tokens[beam_id].tolist()

            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )
            _scores = _scores.view(bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            next_batch_beam = []

            for sent_id in range(bs):

                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )
                    continue

                next_sent_beam = []

                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    beam_id = torch.div(idx, n_words, rounding_mode="trunc")
                    word_id = idx % n_words

                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    if len(next_sent_beam) == beam_size:
                        break

                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0)
                    ] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            cur_len = cur_len + 1

            if all(done):
                break

        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)

        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps, top_k_hash


class LatentBridge(nn.Module):
    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params
        self.dec_emb_dim = self.params.dec_emb_dim
        self.latent_dim = self.params.latent_dim
        self.seq_len_dec = self.params.max_src_len

        self.expand_proj = nn.Linear(self.latent_dim, self.seq_len_dec * self.dec_emb_dim)
        self.relu = nn.ReLU()
        self.refine_proj = nn.Linear(self.dec_emb_dim, self.dec_emb_dim)

    def forward(
        self,
        z_enc,
    ):
        x = self.expand_proj(z_enc)
        x = self.relu(x)
        x = x.view(-1, self.seq_len_dec, self.dec_emb_dim)
        src_enc = self.refine_proj(x)
        
        return src_enc


class LatentBridgeV1(nn.Module):
    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params
        self.dec_emb_dim = self.params.dec_emb_dim
        self.latent_dim = self.params.latent_dim
        self.seq_len_dec = self.params.max_src_len

        
        self.refine_proj = nn.Linear(self.dec_emb_dim, self.dec_emb_dim)

    def forward(
        self,
        z_enc,
    ):
        
        src_enc = self.refine_proj(z_enc)
        
        return src_enc


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        self.max_len = max_len - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.hyp)]
                )
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return (
                self.worst_score
                >= best_sum_logprobs / self.max_len ** self.length_penalty
            )


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    if top_k > 0:
        logits = TopKLogitsWarper(
            top_k=top_k,
            filter_value=filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
        )(None, logits)

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits


class LogitsWarper:

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TopKLogitsWarper(LogitsWarper):

    def __init__(
        self,
        top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {top_k}"
            )

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        top_k = min(
            max(self.top_k, self.min_tokens_to_keep), scores.size(-1)
        )
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopPLogitsWarper(LogitsWarper):

    def __init__(
        self,
        top_p: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

