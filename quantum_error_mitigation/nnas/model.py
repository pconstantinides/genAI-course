"""
Simplest version of the Neural Noise Accumulation Surrogate (NNAS),
matching Fig. 1c / Fig. E1 / Eqs. 5-7 & 19-23 of the paper.

Three stages:
  1. Uniform Neural Accumulator F (vanilla RNN cell, Eq. 19):
         H_l = tanh(W_x X_l + b_x + W_h H_{l-1} + b_h)
  2. Recover low-rank surrogates of U_l, N_l from H_l (Eq. 6/22):
         U_hat_l = W_U H_l + b_U ,   N_hat_l = W_N H_l + b_N
  3. Attention-assisted noise-impact extractor (Eq. 7/23):
         A_l = softmax( (U_hat_l N_hat_l^T) / sqrt(d) ) @ U_hat_l
     followed by a linear readout -> scalar r_hat_l.

Mitigated value (Eq. 2):
     y_em_l = y_tilde_l / ( prod_{j<=l}(1 - p_hat_j) + r_hat_l )
"""

import math
import torch
import torch.nn as nn


class NNASCore(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, d: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d = d

        self.linear_x = nn.Linear(feature_dim, hidden_dim)
        self.linear_h = nn.Linear(hidden_dim, hidden_dim)

        self.W_U = nn.Linear(hidden_dim, d)
        self.W_N = nn.Linear(hidden_dim, d)

        self.readout = nn.Linear(d, 1)
        # Small init on the readout so r starts close to 0 (softplus(0)=log 2),
        # keeping the initial denominator well-conditioned.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, X: torch.Tensor, H0: torch.Tensor = None):
        """
        X: (batch, L, feature_dim)
        returns r: (batch, L)
        """
        batch, L, _ = X.shape
        device = X.device
        H = torch.zeros(batch, self.hidden_dim, device=device) if H0 is None else H0

        r_out = []
        for l in range(L):
            X_l = X[:, l, :]
            H = torch.tanh(self.linear_x(X_l) + self.linear_h(H))

            U_hat = self.W_U(H)  # (batch, d)
            N_hat = self.W_N(H)  # (batch, d)

            outer = torch.bmm(U_hat.unsqueeze(2), N_hat.unsqueeze(1))  # (batch, d, d)
            attn = torch.softmax(outer / math.sqrt(self.d), dim=-1)
            A_l = torch.bmm(attn, U_hat.unsqueeze(2)).squeeze(2)       # (batch, d)

            # softplus enforces r_l >= 0, which guarantees the mitigation
            # denominator (prior + r_l) stays bounded away from zero, since
            # prior = prod_{j<=l}(1-p_hat_j) > 0 whenever all p_hat_j < 1.
            # Without this, an untrained/adversarial r can push (prior + r)
            # arbitrarily close to zero, causing y_em = noisy_y/(prior+r)
            # to explode -- this is a real numerical-stability failure mode
            # of Eq. 2, not just a training artifact.
            r_l = torch.nn.functional.softplus(self.readout(A_l).squeeze(-1))
            r_out.append(r_l)

        return torch.stack(r_out, dim=1)  # (batch, L)


class NNASForQEM(nn.Module):
    """
    Full pipeline: embeds per-layer specification features (optionally
    concatenating the noisy result), runs NNASCore to get r_hat_l, then
    applies the mitigation formula (Eq. 2).
    """

    def __init__(self, spec_dim: int, hidden_dim: int = 32, d: int = 8,
                 use_noisy_results: bool = True):
        super().__init__()
        self.use_noisy_results = use_noisy_results
        feature_dim = spec_dim + (1 if use_noisy_results else 0)
        self.core = NNASCore(feature_dim=feature_dim, hidden_dim=hidden_dim, d=d)

    def forward(self, specs: torch.Tensor, noisy_y: torch.Tensor, p_hat: torch.Tensor):
        """
        specs:   (batch, L, spec_dim)
        noisy_y: (batch, L)
        p_hat:   (batch, L)   per-layer effective rates
        returns  y_em: (batch, L), r: (batch, L)
        """
        if self.use_noisy_results:
            X = torch.cat([specs, noisy_y.unsqueeze(-1)], dim=-1)
        else:
            X = specs

        r = self.core(X)

        prior = torch.cumprod(1.0 - p_hat, dim=1)  # prod_{j<=l}(1 - p_hat_j)
        y_em = noisy_y / (prior + r)
        return y_em, r


# ============================================================================
# DUAL-STATE EXTENSION -- everything below is new, additive; NNASCore and
# NNASForQEM above are untouched (they remain the "Original NNAS" baseline).
#
# Research plan: "Dual-State NNAS with Lie-Algebra Inspired Coherent Error
# Modeling". Replaces the single recurrent hidden state H_l with two coupled
# hidden states (H_l^(s), H_l^(c)):
#   - H^(s): stochastic/decoherent branch. Same vanilla-tanh RNN cell as the
#     original NNASCore (Eq. 19) -- literally "the original NNAS behavior",
#     so that a stochastic-branch-only ablation is architecturally identical
#     to NNASCore (useful as a consistency check, see ablation study).
#   - H^(c): coherent/unitary branch, conceptually approximating Lie-algebra
#     error propagation delta_H_l ~= Ad_{U_l}(delta_h_l) + delta_H_{l-1} --
#     implemented here (per the plan's proof-of-concept allowance) as a
#     GRUCell rather than exact BCH propagation, giving it a genuinely
#     different update rule from the stochastic branch's plain RNN cell.
#
# The two hidden states are concatenated before being fed into an EXACT
# copy of NNASCore's extractor (Eqs. 6/22, 7/23) -- same attention formula,
# same softplus-constrained readout -- only the input dimension changes
# (hidden_dim if only one branch is active, 2*hidden_dim if both are, for
# the ablation study).
# ============================================================================
class DualStateNNASCore(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, d: int = 8,
                 use_stochastic: bool = True, use_coherent: bool = True):
        super().__init__()
        if not (use_stochastic or use_coherent):
            raise ValueError("At least one of use_stochastic/use_coherent must be True.")
        self.hidden_dim = hidden_dim
        self.d = d
        self.use_stochastic = use_stochastic
        self.use_coherent = use_coherent

        # Stochastic branch H^(s): identical cell to NNASCore's RNN (Eq. 19).
        if use_stochastic:
            self.linear_x_s = nn.Linear(feature_dim, hidden_dim)
            self.linear_h_s = nn.Linear(hidden_dim, hidden_dim)

        # Coherent branch H^(c): a GRUCell, per the plan's "proof of concept
        # may simply implement this using a second RNN/GRU" allowance --
        # gives this branch a distinct (gated) update rule from the
        # stochastic branch's plain tanh cell.
        if use_coherent:
            self.gru_c = nn.GRUCell(feature_dim, hidden_dim)

        combined_dim = hidden_dim * (int(use_stochastic) + int(use_coherent))

        # Extractor: identical formula to NNASCore's Step 2/3, just with an
        # input dimension that depends on how many branches are active.
        self.W_U = nn.Linear(combined_dim, d)
        self.W_N = nn.Linear(combined_dim, d)
        self.readout = nn.Linear(d, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, X: torch.Tensor):
        """
        X: (batch, L, feature_dim)
        returns r: (batch, L), and (H_s_seq, H_c_seq) each (batch, L, hidden_dim)
        or None for whichever branch is disabled -- the per-layer hidden
        states are returned too so latent_space_analysis.py can inspect them.
        """
        batch, L, _ = X.shape
        device = X.device

        H_s = torch.zeros(batch, self.hidden_dim, device=device) if self.use_stochastic else None
        H_c = torch.zeros(batch, self.hidden_dim, device=device) if self.use_coherent else None

        r_out = []
        H_s_seq, H_c_seq = [], []
        for l in range(L):
            X_l = X[:, l, :]
            parts = []
            if self.use_stochastic:
                H_s = torch.tanh(self.linear_x_s(X_l) + self.linear_h_s(H_s))
                parts.append(H_s)
                H_s_seq.append(H_s)
            if self.use_coherent:
                H_c = self.gru_c(X_l, H_c)
                parts.append(H_c)
                H_c_seq.append(H_c)

            H_combined = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

            U_hat = self.W_U(H_combined)
            N_hat = self.W_N(H_combined)
            outer = torch.bmm(U_hat.unsqueeze(2), N_hat.unsqueeze(1))
            attn = torch.softmax(outer / math.sqrt(self.d), dim=-1)
            A_l = torch.bmm(attn, U_hat.unsqueeze(2)).squeeze(2)

            r_l = torch.nn.functional.softplus(self.readout(A_l).squeeze(-1))
            r_out.append(r_l)

        r = torch.stack(r_out, dim=1)
        H_s_seq = torch.stack(H_s_seq, dim=1) if H_s_seq else None
        H_c_seq = torch.stack(H_c_seq, dim=1) if H_c_seq else None
        return r, H_s_seq, H_c_seq


class DualStateNNASForQEM(nn.Module):
    """
    Structurally identical wrapper to NNASForQEM (same feature embedding,
    same mitigation formula Eq. 2) -- the only difference is that the core
    accumulator is DualStateNNASCore instead of NNASCore. use_stochastic/
    use_coherent are exposed here too, so this single class covers all four
    ablation configurations (original-equivalent, stochastic-only,
    coherent-only, full dual-state).
    """

    def __init__(self, spec_dim: int, hidden_dim: int = 32, d: int = 8,
                 use_noisy_results: bool = True,
                 use_stochastic: bool = True, use_coherent: bool = True):
        super().__init__()
        self.use_noisy_results = use_noisy_results
        feature_dim = spec_dim + (1 if use_noisy_results else 0)
        self.core = DualStateNNASCore(
            feature_dim=feature_dim, hidden_dim=hidden_dim, d=d,
            use_stochastic=use_stochastic, use_coherent=use_coherent,
        )

    def forward(self, specs: torch.Tensor, noisy_y: torch.Tensor, p_hat: torch.Tensor,
                return_latents: bool = False):
        if self.use_noisy_results:
            X = torch.cat([specs, noisy_y.unsqueeze(-1)], dim=-1)
        else:
            X = specs

        r, H_s_seq, H_c_seq = self.core(X)

        prior = torch.cumprod(1.0 - p_hat, dim=1)
        y_em = noisy_y / (prior + r)

        if return_latents:
            return y_em, r, H_s_seq, H_c_seq
        return y_em, r