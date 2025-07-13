

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn.distributions import NormalParamExtractor

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.linear = nn.Linear(2, d_model)

#     def forward(self, xy):
#         return self.linear(xy)  # learnable x,y embedding
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2, d_model)

    def forward(self, xy):
        """
        xy: Tensor of shape [B, N, 2]
        Normalizes x and y per batch to [-1, 1]
        """
        x = xy[:, :, 0:1]  # [B, N, 1]
        y = xy[:, :, 1:2]  # [B, N, 1]

        x_min = x.min(dim=1, keepdim=True).values  # [B, 1, 1]
        x_max = x.max(dim=1, keepdim=True).values
        y_min = y.min(dim=1, keepdim=True).values
        y_max = y.max(dim=1, keepdim=True).values

        x_range = (x_max - x_min).clamp(min=1e-6)   # [B, 1, 1]
        y_range = (y_max - y_min).clamp(min=1e-6)

        x_norm = (x - x_min) / x_range * 2 - 1      # [B, N, 1]
        y_norm = (y - y_min) / y_range * 2 - 1      # [B, N, 1]

        xy_norm = torch.cat([x_norm, y_norm], dim=-1)  # [B, N, 2]
        return self.linear(xy_norm)                    # [B, N, d_model]

class EnvironmentTransformerOLD(nn.Module):
    
    def __init__(self, num_features, num_heuristics, d_model=128, d_feedforward=2048, num_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_heuristics = num_heuristics

        # Embed features + positional encoding
        self.feature_embed = nn.Linear(num_features, d_model)
        self.pos_embed = PositionalEncoding(d_model)
        
        # Transformer encoder: attends over all environment cells
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder for heuristic weights per robot
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 2*num_heuristics),
            # nn.Softmax(dim=-1)  # softmax over decoder outputs
            nn.Tanh() # hold outputs in [-1, 1] range
        )

        self.norm_extractor = NormalParamExtractor()

        self.calls = 0

    def forward(self, cell_features, cell_positions, robot_positions):
        """
        cell_features: [N, F] #[B, N, F]
        cell_positions: [N, 2] #[B, N, 2]
        robot_positions: [R, 2] #[B, R, 2]
        """
        # print("cell features shape:", cell_features.shape)
        unbatched = False
        if len(cell_features.shape) == 2:
            unbatched = True
            cell_features = cell_features.unsqueeze(0)  # [1, N, F]
            cell_positions = cell_positions.unsqueeze(0)
            robot_positions = robot_positions.unsqueeze(0)
            B, N, F = cell_features.shape
            R = robot_positions.shape[1]

        # Encode features and positions
        x_feat = self.feature_embed(cell_features)             # [B, N, D]
        x_pos = self.pos_embed(cell_positions)                 # [B, N, D]
        x = x_feat + x_pos                                     # [B, N, D]

        # print("x emb shape:", x.shape)

        # === Encoder ===
        enc_out = self.encoder(x)                              # [B, N, D]

        # print("enc_out shape:", enc_out.shape)      # [B, N, D]   

        # === Robot Cell Embedding ===
        # Find closest cell for each robot
        with torch.no_grad():
            # [B, R, N, 2] - [B, 1, N, 2] -> distance to all cells
            cell_pos_exp = cell_positions.unsqueeze(1)         # [B, 1, N, 2]
            robot_pos_exp = robot_positions.unsqueeze(2)       # [B, R, 1, 2]
            # print("cell pos exp shape:", cell_pos_exp.shape)
            # print("robot pos exp shape:", robot_pos_exp.shape)
            dists = torch.norm(robot_pos_exp - cell_pos_exp, dim=-1)  # [B, R, N]
            closest_idxs = torch.argmin(dists, dim=-1)         # [B, R]
            # print("dists shape:", dists.shape)
            # print("closest idxs shape:", closest_idxs.shape)
            # print("closest idxs:", closest_idxs)

        # Gather embeddings of robot-assigned cells
        robot_tokens = torch.gather(
            enc_out, 1,
            closest_idxs.unsqueeze(-1).expand(-1, -1, self.d_model)
        )                                                      # [B, R, D]
        # print("robot tokens shape:", robot_tokens.shape)
        # print("!! Verifying enc tokens:\n", enc_out, "\n robot tokens:\n", robot_tokens)

        # === Decoder ===
        decoder_out = self.decoder(tgt=robot_tokens, memory=enc_out)  # [B, R, D]
        # print("decoder out shape:", decoder_out.shape)

        # === Heuristic output ===
        vals = self.output_head(decoder_out)
        # print("Token mlp vals (softmaxed):\n", vals)
        # print("Token mlp vals shape:\n", vals.shape)
        # h_weights_loc, h_weights_scale = vals[:, :, :vals.shape[-1]//2], vals[:, :, vals.shape[-1]//2:]     # 2*[B, R, num_heuristics]
        # print("h_weights loc:\n", h_weights_loc)
        # print("h_weights scale:\n", h_weights_scale)

        h_weights_loc, h_weights_scale = self.norm_extractor(vals)
        # print("Extracted params:\n", h_weights_loc, h_weights_scale)
        # h_weights_loc, h_weights_scale = self.norm_extractor(vals).chunk(2, dim=-1)
        if self.calls % 1000 == 0:
            print(f"Sample positional enc at call {self.calls}:\n", x_pos)
            print(f"Sample encoder out at call {self.calls}:\n", enc_out)
            print(f"Sample decoder out at call {self.calls}:\n", vals)
            print(f"Sample h_weights loc at call {self.calls}:\n", h_weights_loc)
            print(f"Sample h_weights scale at call {self.calls}:\n", h_weights_scale)

        self.calls += 1

        if unbatched:
            return h_weights_loc[0], h_weights_scale[0] # removes batch dim from actions
        else:
            return h_weights_loc, h_weights_scale 


class EnvironmentTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_heuristics,
        max_robots=8, # < -- With slicing, does not need to match env
        d_model=128,
        d_feedforward=2048,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.2,
        noise_std=0.1,
        max_cells=100, # <-- IMPORTANT: match padding size for env
        cell_pos_as_features=True,
        agent_id_enc=True,
        agent_attn=False,
        variable_team_size=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heuristics = num_heuristics
        self.max_robots = max_robots
        self.noise_std = noise_std
        self.max_cells = max_cells
        self.cell_pos_as_features = cell_pos_as_features
        self.agent_id_enc = agent_id_enc
        self.agent_attn = agent_attn
        self.variable_team_size = variable_team_size

        # Embeddings
        if self.cell_pos_as_features:
            self.feature_embed = nn.Linear(num_features+2, d_model)
        else:
            self.feature_embed = nn.Linear(num_features, d_model)
        self.pos_embed = PositionalEncoding(d_model)
        if agent_id_enc:
            self.agent_embed = nn.Embedding(max_robots, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer decoder (self + cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if self.agent_attn:
            self.agent_cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # Output head: raw logits for Categorical
        self.output_head = nn.Linear(d_model, 2 * num_heuristics)
        self.norm_extractor = NormalParamExtractor()
        self.calls = 0

    # cell_lengths should be a tensor (B,) or (B, 1) indicating the actual number of cells
    def forward(self, cell_features, cell_positions, num_cells, robot_positions, num_robots):
        unbatched = False
        if cell_features.dim() == 2:
            unbatched = True
            cell_features = cell_features.unsqueeze(0)
            cell_positions = cell_positions.unsqueeze(0)
            robot_positions = robot_positions.unsqueeze(0)
            num_cells = num_cells.unsqueeze(0) # Ensure cell_lengths is also batched

        B, N_padded, _ = cell_features.shape # N_padded is MAX_CELLS
        R = robot_positions.size(1)  # R is padded to max_n_agents for env

        # Create attention mask for padding
        # mask is (B, N_padded) boolean tensor, True where token should be ignored (padding)
        # N_padded is self.max_cells
        mask = torch.arange(N_padded, device=cell_features.device).expand(B, N_padded) >= num_cells.unsqueeze(1)
        # Transformer's `src_key_padding_mask` expects True for masked elements.

        # === Encoder input ===
        if self.cell_pos_as_features:
            cell_feats_with_pos = torch.cat([cell_features, cell_positions], dim=-1)
            # print("\nSampled Features:", cell_feats_with_pos[0][:5])
            x_feat = self.feature_embed(cell_feats_with_pos)
            x = x_feat
        else:
            x_feat = self.feature_embed(cell_features)
            x_pos = self.pos_embed(cell_positions) # Use the linear embedding for positions
            x = x_feat + x_pos

        # Pass the mask to the encoder
        enc_out = self.encoder(x, src_key_padding_mask=mask)

        # === Gather robot tokens ===
        with torch.no_grad():
            # Create robot mask: True where robot is real (not padding)
            robot_mask = torch.arange(R, device=robot_positions.device).expand(B, R) < num_robots.unsqueeze(1)  # [B, R]
            
            cell_pos_exp = cell_positions.unsqueeze(1)  # [B, 1, N_padded, 2]
            robot_pos_exp = robot_positions.unsqueeze(2)  # [B, R, 1, 2]
            
            # Compute distances for all robots to all cells
            dists = torch.norm(robot_pos_exp - cell_pos_exp, dim=-1)  # [B, R, N_padded]
            
            # Mask out distances for padded cells
            dists = dists.masked_fill(mask.unsqueeze(1), float('inf'))  # [B, R, N_padded]
            # Mask out distances for padded robots
            dists = dists.masked_fill(~robot_mask.unsqueeze(-1), float('inf'))  # [B, R, N_padded]
            
            closest_idxs = torch.argmin(dists, dim=-1)  # [B, R]
            
            # Check for bad indices (robots assigned to padded cells)
            bad_idx = (mask.gather(1, closest_idxs)).any()
            if bad_idx:
                print("\n!!! ==> closest_idxs includes padding index!")
                print("\tNum cells check:\n\t", num_cells)
                print("\tRobot positions:\n\t", robot_positions)
                print("\tDists:\n\t", dists)

        assert (num_cells > 0).all(), "Zero real cells in one or more batches!"

        # Gather robot tokens, set tokens for padded robots to zero
        robot_tokens = torch.gather(
            enc_out,
            1,
            closest_idxs.unsqueeze(-1).expand(-1, -1, self.d_model)
        )   # [B, R, D]
        robot_tokens = robot_tokens * robot_mask.unsqueeze(-1)  # Zero out tokens for padded robots

        # print("Masked robot tokens:", robot_tokens[:5])

        # === Inject noise ===
        if self.training and self.noise_std > 0:
            robot_tokens = robot_tokens + torch.randn_like(robot_tokens) * self.noise_std * robot_mask.unsqueeze(-1)

        # print("Noised robot tokens:", robot_tokens[:5])

        # === Add positional & agent-ID embeddings ===
        # For padded robots, their positions are meaningless, so mask their embeddings
        robot_pos_enc = self.pos_embed(robot_positions)  # [B, R, D]
        robot_pos_enc = robot_pos_enc * robot_mask.unsqueeze(-1)  # Zero out embeddings for padded robots
        robot_tokens = robot_tokens + robot_pos_enc

        # print("Added position encodings:", robot_tokens[:5])

        if self.agent_id_enc:
            id_indices = torch.arange(self.max_robots, device=robot_tokens.device)[None, :]
            agent_id_enc = self.agent_embed(id_indices).expand(B, -1, -1)
            agent_id_enc = agent_id_enc[:, :R, :]  # Slice to match actual number of robots
            agent_id_enc = agent_id_enc * robot_mask.unsqueeze(-1)  # Zero out embeddings for padded robots
            robot_tokens = robot_tokens + agent_id_enc

        # print("Added id encoding:", robot_tokens[:5])

        # === Decoder ===
        # The `memory_key_padding_mask` tells the decoder to ignore padded elements in `enc_out` (memory)
        decoder_out = self.decoder(tgt=robot_tokens, memory=enc_out, memory_key_padding_mask=mask, tgt_key_padding_mask=robot_mask)
        # print("Decoder out pre-attn:", decoder_out[:5])
        if self.agent_attn:
            cross_agents, _ = self.agent_cross_attn(decoder_out, decoder_out, decoder_out)
            decoder_out = decoder_out + cross_agents

        # === Heuristic outputs ===
        vals = self.output_head(decoder_out)
        vals = vals.reshape(vals.shape[0], -1)
        h_loc, h_scale = self.norm_extractor(vals)

        if self.calls % 8000 == 0:
            # print(f"Sample positional enc at call {self.calls}:\n", x_pos)
            print(f"Sample features at call {self.calls}:\n", cell_feats_with_pos[:5])
            print(f"Sample embedded features at call {self.calls}:\n", x[:5])
            print(f"Sample attention weights at call {self.calls}:\n",)
            print(f"Sample encoder out at call {self.calls}:\n", enc_out[:5])
            print(f"Sample decoder out at call {self.calls}:\n", vals[:5])
            print(f"Sample h_weights loc at call {self.calls}:\n", h_loc[:5])
            print(f"Sample h_weights scale at call {self.calls}:\n", h_scale[:5])

        self.calls += 1

        if unbatched:
            return h_loc[0], h_scale[0]
        return h_loc, h_scale
    

class EnvironmentCriticTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        d_model=128,
        num_heads=4,
        num_layers=2,
        use_attention_pool=True,
        cell_pos_as_features=True,
        max_cells=100 # <-- IMPORTANT: Add this to match your padding size
    ):
        super().__init__()
        self.d_model = d_model
        self.use_attention_pool = use_attention_pool
        self.max_cells = max_cells # Store max_cells
        self.cell_pos_as_features = cell_pos_as_features

        # Embeddings
        if self.cell_pos_as_features:
            self.feature_embed = nn.Linear(num_features+2, d_model)
        else:
            self.feature_embed = nn.Linear(num_features, d_model)
            self.pos_embed = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_attention_pool:
            self.attn_pool = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, d_model))  # learned query

        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    # cell_lengths should be a tensor (B,) or (B, 1) indicating the actual number of cells
    def forward(self, cell_feats, cell_pos, num_cells):
        """
        cell_feats: [B, N_padded, F]
        cell_pos:   [B, N_padded, 2]
        cell_lengths: [B,] or [B, 1] - actual number of cells
        """
        # Ensure cell_lengths is 1D or (B,1) if it's not already
        if num_cells.ndim > 1:
            num_cells = num_cells.squeeze(-1)

        B, N_padded, _ = cell_feats.shape

        # Create attention mask for padding
        mask = torch.arange(N_padded, device=cell_feats.device).expand(B, N_padded) >= num_cells.unsqueeze(1)
        # True means masked (ignored)

         # === Encoder input ===
        if self.cell_pos_as_features:
            cell_feats_with_pos = torch.cat([cell_feats, cell_pos], dim=-1)
            # print("\nSampled Features:", cell_feats_with_pos[0][:5])
            x_feat = self.feature_embed(cell_feats_with_pos)
            x = x_feat
        else:
            x_feat = self.feature_embed(cell_feats)       # [B, N_padded, D]
            x_pos = self.pos_embed(cell_pos) # Use the linear embedding for positions
            x = x_feat + x_pos

        # Pass the mask to the encoder
        enc_out = self.encoder(x, src_key_padding_mask=mask) # [B, N_padded, D]

        if self.use_attention_pool:
            B = enc_out.size(0)
            q = self.query.expand(B, -1, -1)                # [B, 1, D]
            # When using MultiheadAttention, `key_padding_mask` tells it to ignore certain key-value pairs
            # The mask used here should be for the `enc_out` which serves as `key` and `value`.
            attn_out, _ = self.attn_pool(q, enc_out, enc_out, key_padding_mask=mask) # [B, 1, D]
            pooled = attn_out.squeeze(1)                    # [B, D]
        else:
            # If not using attention pool, simple mean pooling over padded elements needs masking
            # Sum valid elements and divide by actual length
            sum_pooled = (enc_out * (~mask).unsqueeze(-1)).sum(dim=1) # Sum only non-masked elements
            # Avoid division by zero if cell_lengths is 0
            pooled = sum_pooled / (num_cells.float().unsqueeze(-1).clamp(min=1e-5)) # Mean of non-masked elements

        value = self.critic_head(pooled)                    # [B, 1]
        return value
    


class EnvironmentTransformer_NoMask(nn.Module):
    def __init__(
        self,
        num_features,
        num_heuristics,
        num_robots=4,
        d_model=128,
        d_feedforward=2048,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.2,
        noise_std=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heuristics = num_heuristics
        self.num_robots = num_robots
        self.noise_std = noise_std

        # Embeddings
        self.feature_embed = nn.Linear(num_features, d_model)
        self.pos_embed = PositionalEncoding(d_model)
        self.agent_embed = nn.Embedding(num_robots, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer decoder (self + cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: raw logits for Categorical
        self.output_head = nn.Linear(d_model, 2 * num_heuristics)
        self.norm_extractor = NormalParamExtractor()
        self.calls = 0

    def forward(self, cell_features, cell_positions, robot_positions):
        unbatched = False
        if cell_features.dim() == 2:
            unbatched = True
            cell_features = cell_features.unsqueeze(0)
            cell_positions = cell_positions.unsqueeze(0)
            robot_positions = robot_positions.unsqueeze(0)

        B, N, _ = cell_features.shape
        R = robot_positions.size(1)

        # === Encoder input ===
        x_feat = self.feature_embed(cell_features)
        x_pos = self.pos_embed(cell_positions)
        x = x_feat + x_pos
        enc_out = self.encoder(x)

        # === Gather robot tokens ===
        with torch.no_grad():
            cell_pos_exp = cell_positions.unsqueeze(1)
            robot_pos_exp = robot_positions.unsqueeze(2)
            dists = torch.norm(robot_pos_exp - cell_pos_exp, dim=-1)
            closest_idxs = torch.argmin(dists, dim=-1)

        robot_tokens = torch.gather(
            enc_out,
            1,
            closest_idxs.unsqueeze(-1).expand(-1, -1, self.d_model)
        )  # [B, R, D]

        # === Inject noise ===
        if self.training and self.noise_std > 0:
            robot_tokens = robot_tokens + torch.randn_like(robot_tokens) * self.noise_std

        # === Add positional & agent-ID embeddings ===
        # robot_positions: [B, R, 2]
        robot_pos_enc = self.pos_embed(robot_positions)
        id_indices = torch.arange(R, device=robot_tokens.device)[None, :]
        agent_id_enc = self.agent_embed(id_indices).expand(B, -1, -1)
        robot_tokens = robot_tokens + robot_pos_enc + agent_id_enc

        # === Decoder ===
        decoder_out = self.decoder(tgt=robot_tokens, memory=enc_out)

        # === Heuristic outputs ===
        vals = self.output_head(decoder_out)
        h_loc, h_scale = self.norm_extractor(vals)

        self.calls += 1
        if self.calls % 1000 == 0:
            print(f"Sample positional enc at call {self.calls}:\n", x_pos)
            print(f"Sample encoder out at call {self.calls}:\n", enc_out)
            print(f"Sample decoder out at call {self.calls}:\n", vals)
            print(f"Sample h_weights loc at call {self.calls}:\n", h_loc)
            print(f"Sample h_weights scale at call {self.calls}:\n", h_scale)

        if unbatched:
            return h_loc[0], h_scale[0]
        return h_loc, h_scale
    

class EnvironmentCriticTransformer_NoMask(nn.Module):
    def __init__(self, num_features, d_model=128, num_heads=4, num_layers=2, use_attention_pool=True):
        super().__init__()
        self.d_model = d_model
        self.use_attention_pool = use_attention_pool

        self.feature_embed = nn.Linear(num_features, d_model)
        self.pos_embed = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_attention_pool:
            self.attn_pool = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, d_model))  # learned query

        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, cell_feats, cell_pos):
        """
        cell_feats: [B, N, F]
        cell_pos:   [B, N, 2]
        """
        feat_emb = self.feature_embed(cell_feats)      # [B, N, D]
        pos_emb = self.pos_embed(cell_pos)             # [B, N, D]
        x = feat_emb + pos_emb                         # [B, N, D]

        enc_out = self.encoder(x)                      # [B, N, D]

        if self.use_attention_pool:
            B = enc_out.size(0)
            q = self.query.expand(B, -1, -1)           # [B, 1, D]
            attn_out, _ = self.attn_pool(q, enc_out, enc_out)  # [B, 1, D]
            pooled = attn_out.squeeze(1)               # [B, D]
        else:
            pooled = enc_out.mean(dim=1)               # [B, D]

        value = self.critic_head(pooled)               # [B, 1]
        return value