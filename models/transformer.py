

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn.distributions import NormalParamExtractor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2, d_model)

    def forward(self, xy):
        return self.linear(xy)  # learnable x,y embedding

class EnvironmentTransformer(nn.Module):
    def __init__(self, num_features, num_robots, num_heuristics, d_model=128, num_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_robots = num_robots
        self.num_heuristics = num_heuristics

        # Embed features + positional encoding
        self.feature_embed = nn.Linear(num_features, d_model)
        self.pos_embed = PositionalEncoding(d_model)
        
        # Transformer encoder: attends over all environment cells
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Critic: global value prediction from attended environment
        self.critic_head = nn.Sequential(
            nn.Linear(16, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

        # Decoder for heuristic weights per robot
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, 2*num_heuristics)

        self.norm_extractor = NormalParamExtractor()

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

        # print("enc_out shape:", enc_out.shape)

        # === Critic ===
        global_embedding = enc_out.mean(dim=-1)                 # [B, D]
        # print("global embedding shape:", global_embedding.shape)
        value = self.critic_head(global_embedding)
        # print("value shape:", value.shape)

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
        # print("Token mlp vals:\n", vals)
        # print("Token mlp vals shape:\n", vals.shape)
        # h_weights_loc, h_weights_scale = vals[:, :, :vals.shape[-1]//2], vals[:, :, vals.shape[-1]//2:]     # 2*[B, R, num_heuristics]
        # print("h_weights loc:\n", h_weights_loc)
        # print("h_weights scale:\n", h_weights_scale)

        h_weights_loc, h_weights_scale = self.norm_extractor(vals)
        # print("Extracted params:\n", extracted_params)
        # h_weights_loc, h_weights_scale = self.norm_extractor(vals).chunk(2, dim=-1)

        if unbatched:
            return value, h_weights_loc[0], h_weights_scale[0] # removes batch dim from actions
        else:
            return value, h_weights_loc, h_weights_scale 
