

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(2, embed_dim)

    def forward(self, xy):
        return self.linear(xy)  # learnable x,y embedding

class EnvironmentTransformer(nn.Module):
    def __init__(self, num_features, num_robots, num_heuristics, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_robots = num_robots
        self.num_heuristics = num_heuristics

        # Embed features + positional encoding
        self.feature_embed = nn.Linear(num_features, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim)
        
        # Transformer encoder: attends over all environment cells
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Critic: global value prediction from attended environment
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # Decoder for heuristic weights per robot
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(embed_dim, num_heuristics)

    def forward(self, cell_features, cell_positions, robot_positions):
        """
        cell_features: [B, N, F]
        cell_positions: [B, N, 2]
        robot_positions: [B, R, 2]
        """
        B, N, F = cell_features.shape
        R = robot_positions.shape[1]

        # Encode features and positions
        x_feat = self.feature_embed(cell_features)              # [B, N, D]
        x_pos = self.pos_embed(cell_positions)                 # [B, N, D]
        x = x_feat + x_pos                                     # [B, N, D]

        # === Encoder ===
        enc_out = self.encoder(x)                              # [B, N, D]

        # === Critic ===
        global_embedding = enc_out.mean(dim=1)                 # [B, D]
        value = self.critic_head(global_embedding)             # [B, 1]

        # === Robot Cell Embedding ===
        # Find closest cell for each robot
        with torch.no_grad():
            # [B, R, N, 2] - [B, 1, N, 2] -> distance to all cells
            cell_pos_exp = cell_positions.unsqueeze(1)         # [B, 1, N, 2]
            robot_pos_exp = robot_positions.unsqueeze(2)       # [B, R, 1, 2]
            dists = torch.norm(robot_pos_exp - cell_pos_exp, dim=-1)  # [B, R, N]
            closest_idxs = torch.argmin(dists, dim=-1)         # [B, R]

        # Gather embeddings of robot-assigned cells
        robot_tokens = torch.gather(
            enc_out, 1,
            closest_idxs.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )                                                      # [B, R, D]

        # === Decoder ===
        decoder_out = self.decoder(tgt=robot_tokens, memory=enc_out)  # [B, R, D]

        # === Heuristic output ===
        heuristic_weights = self.output_head(decoder_out)      # [B, R, num_heuristics]

        return value.squeeze(-1), heuristic_weights  # [B], [B, R, H]
