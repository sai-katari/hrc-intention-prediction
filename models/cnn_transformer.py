import math
import torch
import torch.nn as nn


class ConvStem(nn.Module):
    """
    1D-CNN feature extractor for a single modality.
    Input (B, T, C_in) -> (B, T, d_model).

    Uses left-only (causal) padding so the representation at timestep t
    only depends on frames up to and including t.
    """

    def __init__(self, in_channels, d_model, kernel_sizes=(5, 5, 3)):
        super().__init__()
        chans = [in_channels] + [d_model] * len(kernel_sizes)
        layers = []
        for i, k in enumerate(kernel_sizes):
            layers += [
                nn.ConstantPad1d((k - 1, 0), 0),
                nn.Conv1d(chans[i], chans[i + 1], kernel_size=k, padding=0),
                nn.BatchNorm1d(chans[i + 1]),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class GoalRelativeFeatures(nn.Module):
    """
    Per-timestep gaze alignment and wrist proximity to each candidate object.
    Output: (B, T, n_goals * 2).

    wrist_xyz is passed explicitly rather than sliced from motion because
    real MoGaze motion is joint angles, not positions — wrist XYZ comes
    from forward kinematics and is computed separately.
    """

    def forward(self, wrist_xyz, gaze, goal_positions):
        gaze_n = gaze / (gaze.norm(dim=-1, keepdim=True) + 1e-6)
        goals_exp = goal_positions.unsqueeze(1)                         # (B, 1, n_goals, 3)
        to_goal = goals_exp - wrist_xyz.unsqueeze(2)                    # (B, T, n_goals, 3)
        to_goal_n = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)
        gaze_align = (gaze_n.unsqueeze(2) * to_goal_n).sum(-1)         # (B, T, n_goals)
        proximity = -to_goal.norm(dim=-1)                               # (B, T, n_goals)
        return torch.cat([gaze_align, proximity], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class IntentionTransformer(nn.Module):
    """
    CNN-Transformer for anytime intention prediction.

    Separate CNN stems process motion and gaze; a goal-relative feature
    branch adds per-timestep gaze alignment and wrist proximity to each
    candidate object. All three are fused and passed through a causal
    Transformer encoder that produces per-timestep logits — so you can
    query predictions at any observation fraction, not just at the end.

    motion_dim: width of the joint-angle vector (66 for MoGaze, 63 for synthetic).
    n_goals: number of candidate objects (10 for MoGaze, 6 for synthetic).
    """

    def __init__(self, motion_dim=63, gaze_dim=3, n_goals=10,
                 d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_goals = n_goals

        self.motion_stem = ConvStem(motion_dim, d_model)
        self.gaze_stem = ConvStem(gaze_dim, d_model)
        self.goal_features = GoalRelativeFeatures()

        fused_dim = d_model * 2 + n_goals * 2
        self.fuse_proj = nn.Linear(fused_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_goals)

    @staticmethod
    def causal_mask(T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, motion, gaze, goal_positions, wrist_xyz, use_goal_features=True):
        """
        Returns logits (B, T, n_goals) — one prediction per timestep.

        use_goal_features=False zeros the goal-relative branch without
        touching the gaze CNN, giving a clean Motion+Gaze ablation.
        """
        m = self.motion_stem(motion)
        g = self.gaze_stem(gaze)

        if use_goal_features:
            gr = self.goal_features(wrist_xyz, gaze, goal_positions)
        else:
            B, T = motion.shape[:2]
            gr = torch.zeros(B, T, self.n_goals * 2, device=motion.device)

        fused = self.fuse_proj(torch.cat([m, g, gr], dim=-1))
        fused = self.pos_enc(fused)
        mask = self.causal_mask(fused.size(1), fused.device)
        return self.classifier(self.encoder(fused, mask=mask))


if __name__ == "__main__":
    B, T = 4, 90
    model = IntentionTransformer(motion_dim=63, n_goals=6)
    out = model(
        torch.randn(B, T, 63),
        torch.randn(B, T, 3),
        torch.randn(B, 6, 3),
        torch.randn(B, T, 3),
    )
    print("output shape:", out.shape)  # (4, 90, 6)