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

class GenerativeCoherentBranch(nn.Module):
    """
    Per layer l:
      Task 2: recurrent posterior q(delta_h_l | x_l, g_l, Delta_H_{l-1}) --
              mu_l, logvar_l. The GRU input is [x_l ; proj(Delta_H_{l-1})],
              closing the feedback loop: the generator sees the physically
              accumulated coherent error so far, not just the raw circuit
              features -- without this, the generator would have no
              knowledge of the coherent error accumulated so far.
      Task 3: gate-conditioned prior p(delta_h_l | g_l) -- depends on g_l
              only (an MLP, no recurrence).
      Task 4: reparameterized sample delta_h_l = mu_l + sigma_l * eps.
      Task 5: deterministic, non-trainable BCH-inspired propagation,
              Delta_H_l = Delta_H_{l-1} + Ad_{U_l}(delta_h_l), first order.

    This codebase has no separate per-layer "gate descriptor" (gate type /
    qubits / parameters) distinct from the embedded circuit features -- per
    the plan's allowance to approximate when exact structure is
    unavailable, g_l is taken to be the same embedded feature vector x_l
    (minimal-change choice; a real gate encoding could replace this
    without touching anything else). Similarly, exact per-layer gate
    unitaries U_l aren't available at this level of the model, so Ad_{U_l}
    is approximated (as the plan explicitly allows) by a single fixed,
    non-trainable orthogonal linear map shared across layers.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Task 2: recurrent posterior. Input is [x_l ; proj(Delta_H_{l-1})]
        # (closed feedback loop, see forward()), so the GRU takes
        # feature_dim + hidden_dim, not just feature_dim.
        self.rnn_cell = nn.GRUCell(feature_dim + hidden_dim, hidden_dim)
        self.coherent_proj = nn.Linear(hidden_dim, hidden_dim)  # projects Delta_H before feedback
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

        # Task 3: gate-conditioned prior (function of g_l only, no recurrence)
        self.prior_mlp = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.prior_mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.prior_logvar_head = nn.Linear(hidden_dim, hidden_dim)

        # Task 5: fixed (non-trainable) BCH-inspired propagation operator Ad_U
        Q, _ = torch.linalg.qr(torch.randn(hidden_dim, hidden_dim))
        self.register_buffer("Ad_U", Q)

    def forward(self, X: torch.Tensor):
        """
        X: (batch, L, feature_dim), used as both x_l and g_l (see docstring).
        Returns:
            Delta_H_seq: (batch, L, hidden_dim) -- accumulated generator per layer
            kl_seq:      (batch, L)             -- per-layer KL(q || prior)
        """
        batch, L, _ = X.shape
        device = X.device
        h = torch.zeros(batch, self.hidden_dim, device=device)
        Delta_H = torch.zeros(batch, self.hidden_dim, device=device)

        Delta_H_out, kl_out = [], []
        for l in range(L):
            x_l = X[:, l, :]
            g_l = x_l  # gate-descriptor proxy (see class docstring)

            # Task 2: closed physical feedback loop -- Delta_H here is still
            # Delta_H_{l-1} (this step's accumulation happens below), so the
            # generator sees exactly what physics has accumulated so far,
            # not what it's about to generate.
            gru_input = torch.cat([x_l, self.coherent_proj(Delta_H)], dim=-1)
            h = self.rnn_cell(gru_input, h)
            mu_l = self.mu_head(h)
            logvar_l = self.logvar_head(h)

            # Task 3
            prior_h = self.prior_mlp(g_l)
            mu_prior = self.prior_mu_head(prior_h)
            logvar_prior = self.prior_logvar_head(prior_h)

            # Task 4: reparameterization trick
            sigma_l = torch.exp(0.5 * logvar_l)
            eps = torch.randn_like(sigma_l)
            delta_h_l = mu_l + sigma_l * eps

            # Task 5: deterministic, first-order BCH-inspired propagation
            Delta_H = Delta_H + delta_h_l @ self.Ad_U.T

            # Closed-form KL between diagonal Gaussians q and prior
            kl_l = 0.5 * (
                (logvar_prior - logvar_l)
                + (logvar_l.exp() + (mu_l - mu_prior) ** 2) / logvar_prior.exp()
                - 1.0
            ).sum(dim=-1)

            Delta_H_out.append(Delta_H)
            kl_out.append(kl_l)

        return torch.stack(Delta_H_out, dim=1), torch.stack(kl_out, dim=1)


class PhysicsInformedNNASCore(nn.Module):
    """Task 1 + 6: unchanged stochastic branch (identical cell to
    NNASCore's), fused via concatenation with a (linearly projected)
    GenerativeCoherentBranch output, fed through an unchanged copy of the
    extractor formula (Eqs. 6/22, 7/23)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32, d: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d = d

        # Task 1: stochastic branch, unchanged (same cell as NNASCore)
        self.linear_x_s = nn.Linear(feature_dim, hidden_dim)
        self.linear_h_s = nn.Linear(hidden_dim, hidden_dim)

        # Task 2-5: generative coherent branch
        self.coherent = GenerativeCoherentBranch(feature_dim, hidden_dim)
        # Task 6: projection of Delta_H into the hidden dimension before fusion
        self.coherent_proj = nn.Linear(hidden_dim, hidden_dim)

        combined_dim = hidden_dim * 2
        self.W_U = nn.Linear(combined_dim, d)
        self.W_N = nn.Linear(combined_dim, d)
        self.readout = nn.Linear(d, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, X: torch.Tensor):
        batch, L, _ = X.shape
        device = X.device
        H_s = torch.zeros(batch, self.hidden_dim, device=device)

        Delta_H_seq, kl_seq = self.coherent(X)  # (batch, L, hidden_dim), (batch, L)

        r_out = []
        for l in range(L):
            X_l = X[:, l, :]
            H_s = torch.tanh(self.linear_x_s(X_l) + self.linear_h_s(H_s))
            H_c = self.coherent_proj(Delta_H_seq[:, l, :])

            H_combined = torch.cat([H_s, H_c], dim=-1)  # Task 6: fusion

            U_hat = self.W_U(H_combined)
            N_hat = self.W_N(H_combined)
            outer = torch.bmm(U_hat.unsqueeze(2), N_hat.unsqueeze(1))
            attn = torch.softmax(outer / math.sqrt(self.d), dim=-1)
            A_l = torch.bmm(attn, U_hat.unsqueeze(2)).squeeze(2)

            r_l = torch.nn.functional.softplus(self.readout(A_l).squeeze(-1))
            r_out.append(r_l)

        r = torch.stack(r_out, dim=1)
        return r, kl_seq


class PhysicsInformedNNASForQEM(nn.Module):
    """Same feature embedding and mitigation formula (Eq. 2) as NNASForQEM/
    DualStateNNASForQEM -- forward returns (y_em, r, kl_seq); the extra
    kl_seq is the Task 7 regularization term (D_KL(q||prior) per layer),
    to be combined into the training loss as L_mitigation + beta*L_KL."""

    def __init__(self, spec_dim: int, hidden_dim: int = 32, d: int = 8,
                 use_noisy_results: bool = True):
        super().__init__()
        self.use_noisy_results = use_noisy_results
        feature_dim = spec_dim + (1 if use_noisy_results else 0)
        self.core = PhysicsInformedNNASCore(feature_dim=feature_dim, hidden_dim=hidden_dim, d=d)

    def forward(self, specs: torch.Tensor, noisy_y: torch.Tensor, p_hat: torch.Tensor):
        if self.use_noisy_results:
            X = torch.cat([specs, noisy_y.unsqueeze(-1)], dim=-1)
        else:
            X = specs

        r, kl_seq = self.core(X)
        prior = torch.cumprod(1.0 - p_hat, dim=1)
        y_em = noisy_y / (prior + r)
        return y_em, r, kl_seq