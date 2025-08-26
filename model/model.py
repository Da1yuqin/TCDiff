from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F, ModuleList, Linear, Module

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
import random

class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(2, 3))    

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

# Adopted from TBIFormer. Since we do not pass it in, this operates the same as standard Attention.
# Multi-head self-attention layer integrated with trajectory information, 
# Trajectory attention is only applied when trj_dist is provided. 
class SBI_MSA(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1, dk = 64):
        super().__init__()
        self.n_head = n_head
        self.d_k = dk # Projection dimension used for multi-head attention

        self.w_qs = nn.Linear(d_model, n_head * dk, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * dk, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * dk, bias=False)

        self.fc = nn.Linear(n_head * dk , d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=np.power(dk, 0.5))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.temperature = self.d_k ** 0.5

    def forward(self, q, k, v, shared_emb, trj_dist = None):

        query, key, value = q, k, v
        d_k, d_v, n_head = self.d_k, self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        
        # projection & reshape: multi-head 
        q = self.w_qs(query).view(sz_b, len_q, n_head, d_k).transpose(1, 2) 
        k = self.w_ks(key).view(sz_b, len_k, n_head, d_k).transpose(1, 2) 
        v = self.w_vs(value).view(sz_b, len_v, n_head, d_v).transpose(1, 2) 

        query_emb = shared_emb.view(1, 10, n_head, d_k).transpose(1, 2) 
        indexed_matrix = torch.matmul(q, query_emb.transpose(2, 3)) 
        # Shape: (sz_b, n_head, len_q, 10)
        # Equivalent to performing an embedding lookup using q as input

        
        # traj_embedding
        trpe_bias = 0
        if trj_dist is not None: 
            # Gather trajectory-aware positional encodings based on trj_dist indices
            trpe_bias = torch.gather( 
                indexed_matrix, 3, trj_dist.unsqueeze(1).repeat(1, n_head, 1, 1)  
            ) # torch.gather replaces values along dim=3 according to indices from trj_dist
            
        # Compute attention scores and integrate TRPE bias (0.)
        attn_output_weights = torch.matmul(q / self.temperature, k.transpose(2, 3)) + trpe_bias 
        attn = self.dropout(F.softmax(attn_output_weights, dim=-1))  # attention score 

        # Apply attention to the value vectors
        output = torch.matmul(attn, v) 
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))

        # Apply layer normalization to stabilize training
        output = self.layer_norm(output) 
        return output

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.pe = nn.Parameter(torch.randn(480, 480))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k=None, v=None):
        b, n, _, h = *q.shape, self.heads
        k = q if k is None else k
        v = q if v is None else v

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        _, _, dots_w, dots_h = dots.shape
        dots += self.pe[:dots_w, :dots_h]
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DenseFiLM(nn.Module): 
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1) 
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention( 
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x # use
        x = self.self_attn( # torch.nn
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    ''' Transformer decoder layer with FiLM conditioning and trajectory-based modulation.'''
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        context_dim = 512,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()

        # Self-attention and cross-attention modules
        self.self_attn = SBI_MSA(nhead, d_model, dropout=dropout)
        self.multihead_attn = SBI_MSA(nhead, d_model, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        # FiLM conditioning layers (DenseFiLM) to modulate features based on t
        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

        # Additional linear transformation and normalization for modulation
        self.linear3 = nn.Linear(d_model, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Trajectory modulation network: 3-layer MLP modulated by context
        # Each layer performs conditional computation with context embedding
        self.traj_Modulation = ModuleList([
            ConcatSquashLinear(d_model, 128, context_dim), 
            ConcatSquashLinear(128, 128, context_dim),
            ConcatSquashLinear(128, d_model, context_dim),
        ])
        self.act = F.leaky_relu


    def forward(
        self,
        tgt,        # x
        memory,     # cond
        t,          # t_emb
        traj_emb,   # Trajectory embedding
        emb,        # Embedding table for trajectory
        trj_dist,   # Trajectory distance information
        tgt_mask=None, 
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt 

        if self.norm_first: # True
            # --- Self-attention block ---
            # Apply LayerNorm → self-attention → FiLM modulation → residual connection
            x_1 = self._sa_block(self.norm1(x), emb, trj_dist=trj_dist) 
            x = x + featurewise_affine(x_1, self.film1(t)) 

            # --- Cross-attention block ---
            # Apply LayerNorm → cross-attention → FiLM modulation → residual connection
            x_2 = self._mha_block( 
                self.norm2(x), memory, emb, trj_dist=trj_dist
            )
            x = x + featurewise_affine(x_2, self.film2(t)) 

            # --- Feedforward block ---
            # Apply LayerNorm → feedforward → FiLM modulation → residual connection
            x_3 = self._ff_block(self.norm3(x)) # mlp + dropout
            x = x + featurewise_affine(x_3, self.film3(t)) 

            # --- Additional trajectory-based modulation ---
            # Prepare input for modulation:
            # - Apply LayerNorm and linear transformation
            x = self.linear3(self.norm4(x))

            # Concatenate temporal embedding with trajectory embeddings as context
            # ctx_emb shape: (B, 1 + seq_len, latent_dim)
            ctx_emb = torch.cat([t.unsqueeze(1), traj_emb], dim=-2) 

            # Pass through the 3-layer trajectory modulation network
            out = x
            for i, layer in enumerate(self.traj_Modulation):
                out = layer(ctx=ctx_emb, x=out)
                if i < len(self.traj_Modulation) - 1:
                    out = self.act(out)
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, ), self.film1(t) 
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    def _sa_block(self, x, shared_emb, trj_dist=None):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            shared_emb,
            trj_dist = trj_dist,
        )
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, shared_emb, trj_dist=None):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x # embedding
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            shared_emb,
            trj_dist = trj_dist,
        )
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))) 
        return self.dropout3(x)



class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t, traj_emb, emb, trj_dist):
        for layer in self.stack:
            x = layer(x, cond, t, traj_emb, emb, trj_dist)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        required_dancer_num = 4,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats
        self.nfeats = nfeats
        self.latent_dim = latent_dim

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )
        self.required_dancer_num = required_dancer_num
        self.seq_len = seq_len
        

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        # self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.cond_projection = nn.Sequential(  # music-condition only, currently
            nn.Linear(cond_feature_dim * 2, cond_feature_dim),
            nn.ReLU(),
            nn.Linear(cond_feature_dim, latent_dim),
        )

        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer( 
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack) 
         
        self.final_layer = nn.Linear(latent_dim, output_feats)

        # Fusion Projection
        self.relative_projection_layer = nn.Sequential( 
            nn.Linear(latent_dim * self.required_dancer_num, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * self.required_dancer_num)
        )
        # self.relative_embedding_layer = nn.Linear(latent_dim * 3, latent_dim * 3) # also fine

        # traj embedding
        self.d_k = 64 
        self.embeddings_table = nn.Embedding(10,  self.d_k* num_heads) # 10 (maximum number of dancers), k_len, head_num 
        
        # traj_embedding
        self.traj_embedding = nn.Sequential(
                            nn.Linear(2, 64),
                            nn.ReLU(),
                            nn.Linear(64, latent_dim), 
                        )

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1) 
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0) 

        return unc + (conditioned - unc) * guidance_weight # guidance_weight == 2

    def forward( 
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0, trj_dist = None 
    ):
        batch_size, device = x.shape[0], x.device

        x = x.reshape(batch_size, -1, 151) 
        # (b, seq_len*dn, 151)

        # xz offset
        traj_emb = self.traj_embedding(x[:,1:,[4,4+1]] - x[:,:-1,[4,4+1]])  

        # project to latent space
        x = self.input_projection(x)  # ([bs, seq_len*dn, nfeats])) -> torch.Size([bs, seq_len*dn, latent])
        x = self.relative_projection_layer(x.reshape(batch_size, self.seq_len, self.latent_dim*self.required_dancer_num)).reshape(batch_size, self.required_dancer_num*self.seq_len, self.latent_dim)
        
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device) # torch.Size([bs])
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1") # torch.Size([bs, 1, 1])
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1") # torch.Size([bs, 1])

        # Music and motion lengths differ (~2x), so adjust music condition sequence length for FPS alignment.
        c_bs,cond_seq_len,_ = cond_embed.shape # (bs, 301, 438)
        if cond_seq_len % 2 == 1: # If odd length
            cond_embed = cond_embed[:,:-1,:].reshape(c_bs, cond_seq_len//2,-1)
        else:
            cond_embed = cond_embed.reshape(c_bs, cond_seq_len//2,-1)

        cond_tokens = self.cond_projection(cond_embed.float()) # Shape: [bs, seq_len, 876] → [bs, seq_len, 512 (latent_dim)]
        # Encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens) # Shape: [bs, seq_len, 512]
        cond_tokens = self.cond_encoder(cond_tokens) # Shape: [bs, seq_len, 512]
        
        # Prepare null conditioning embedding.
        # Randomly drop all cond_embeddings for some batches to encourage the model to rely on other information.
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        # Shape: [1, seq_len, 512]

        # Replace cond_tokens with null_cond_embed in selected batches.
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed) 
        # Shapes: [bs, 1, 1], [bs, seq_len, 512], [1, seq_len, 512]
        
        # Aggregate token information via average pooling.
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2) 
        # Shape: [bs, seq_len, 512] → [bs, 512]

        # Project pooled condition features for use in FiLM modulation.
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens) 
        # Shape: [bs, 512] → [bs, 512]

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times) # [*(bs)]->[*(bs), 2048]

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden) # [*, 2048] -> [*, 512](t)
        t_tokens = self.to_time_tokens(t_hidden) # [*, 2048] -> [*(bs), 2, 512]（t_tokens）

        # FiLM conditioning | Performs concatenation of the input t and cond_embedding.
        # A portion of cond is randomly dropped to encourage the model not to overly rely on it.
        null_cond_hidden = self.null_cond_hidden.to(t.dtype) # Shape: [1, 512]
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden) # Replace cond_hidden with null_cond_hidden for some batches,
                                                                                   # promoting robustness by forcing the model to rely on other information.
        t += cond_hidden # [2, 512] + [2, 512] → [2, 512]

        # Cross-attention conditioning | Embedding shown in the orange box of Figure 2
        c = torch.cat((cond_tokens, t_tokens), dim=-2) # [bs, seq_len*dn, 512] con [bs, 2, 512] -> [bs, seq_len*dn + 2, 512]
        cond_tokens = self.norm_cond(c) # # LayerNorm over concatenated tokens

        # Previously, only input preparation was performed.
        # Now begin the three DecoderBlocks (corresponding to Figure 2 right structure).
        # Pass data through the Transformer decoder, attending to conditional embeddings.
        output = self.seqTransDecoder(x, cond_tokens, t, traj_emb, self.embeddings_table.weight, trj_dist)
        
        output = self.final_layer(output) # -> SMPL 151
        return output
