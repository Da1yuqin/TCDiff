import torch
import numpy as np
import random
import torch.nn as nn
import math
from torch.nn import functional as F
from model.utils import PositionalEncoding



class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=120, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=120, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class music2traj_Transformer(nn.Module):

    def __init__(self, 
                embed_dim=64, 
                music_dim=64, 
                block_size=60, # Sequence length required for the transformer.
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.cond_emb = nn.Linear(music_dim, embed_dim) 
        self.traj_emb = nn.Linear(3, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim + music_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)]) # casual attention mask
        self.pos_embed = PositionalEncoding(embed_dim, drop_out_rate, batch_first=True)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, traj_feat, music_feat):
        # Pass inputs through the Transformer model

        # Apply positional encoding to the trajectory features
        traj_feat = self.pos_embed(traj_feat)

        # Encode the music features
        music_feat = self.cond_emb(music_feat)

        # Compute dancer repetition factor
        dn = traj_feat.shape[1]//music_feat.shape[1]

        # Repeat music features for each dancer's sequence segment
        music_feat = music_feat.repeat(1, dn, 1) 

        # Concatenate music and trajectory features
        x = torch.cat([music_feat, traj_feat], dim=2) 

        x = self.blocks(x)

        return x


class TrajDecoder(nn.Module):
    def __init__(self, 
                 nfeats,
                 trans_layer = 4,
                 window_size = 60,
                 latent_dim: int = 64,
                 dropout: float = 0.1,
                 n_head: int = 4,
                 cond_feature_dim: int = 438, 
                 ):
        super().__init__()
    
        self.latent_dim = latent_dim

        self.lstm = torch.nn.LSTM(input_size = nfeats, hidden_size = latent_dim, num_layers = 3)
        
        # Music feat Extractor
        self.music_projection = nn.Sequential(  
                nn.Linear(cond_feature_dim*2, cond_feature_dim),
                nn.LeakyReLU(),
                nn.Linear(cond_feature_dim, cond_feature_dim),
                nn.LeakyReLU(),
                nn.Linear(cond_feature_dim, latent_dim),
            )

        # Encoder: feature extraction
        self.trans_extractor = music2traj_Transformer(embed_dim = latent_dim,
                                                      drop_out_rate=dropout, 
                                                      block_size= window_size,
                                                      n_head=n_head, 
                                                      music_dim = latent_dim, 
                                                      num_layers=trans_layer) 

        # Decoder
        self.Decoder = nn.Sequential(  
                nn.Linear(latent_dim*3, latent_dim*2),
                nn.LeakyReLU(),
                nn.Linear(latent_dim*2, latent_dim*2),
                nn.LeakyReLU(),
                 nn.Linear(latent_dim*2, latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, nfeats),
            )


    def forward(self, x, music_feat):
        b, dn, seq, c = x.shape
        x = x.reshape(b, dn*seq, c)

        x = self.lstm(x)[0] 

        # Align the sequence length of music features with motion
        c_bs, cond_seq_len, _ = music_feat.shape
        if cond_seq_len % 2 == 1:  # If odd, drop the last frame and reshape
            music_feat = music_feat[:,:-1,:].reshape(c_bs, cond_seq_len//2,-1)
        else: # If even, reshape directly
            music_feat = music_feat.reshape(c_bs, cond_seq_len//2,-1)

        # Project music features to latent space
        music_feat = self.music_projection(music_feat) 

        # Extract joint features from motion and music
        feat = self.trans_extractor(x, music_feat[:,:seq]) 

        # Prepare music features for prediction part
        pred_music_feat = music_feat[:,-seq:] 
        pred_music_feat = pred_music_feat.repeat(1, dn, 1) 

        # Concatenate features for decoding
        feat = torch.cat([feat,pred_music_feat], dim = 2) 

        # Generate predicted motion
        x_pre = self.Decoder(feat) 

        # Reshape back to original multi-dancer format
        return x_pre.reshape(b,dn,seq,c)

