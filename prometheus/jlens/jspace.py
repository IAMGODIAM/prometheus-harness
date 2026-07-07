"""J-Space — Sparse non-negative decomposition in J-lens vector space.

J-space is the set of activations expressible as a sparse non-negative combination
of k ≤ 25 J-lens vectors, solved via gradient pursuit. The J-space component typically
accounts for only 6-10% of activation variance but carries most causal weight for
downstream multi-hop reasoning and creative generation.

The sparse decomposition algorithm is gradient pursuit — a greedy pursuit variant
that supports non-negativity constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class JSpaceResult:
    """Result of J-space decomposition.

    Attributes:
        coefficients: Non-negative sparse coefficients, shape (k,).
        indices: Token indices of the selected J-lens vectors, shape (k,).
        jspace_component: The J-space projection of the activation, shape (d_model,).
        residual: The non-J-space component, shape (d_model,).
        variance_explained: Fraction of activation variance in J-space.
        top_tokens: List of (token_idx, coefficient) tuples sorted by coefficient.
    """

    coefficients: torch.Tensor
    indices: torch.Tensor
    jspace_component: torch.Tensor
    residual: torch.Tensor
    variance_explained: float
    top_tokens: list[tuple[int, float]]


class JSpaceDecomposer:
    """Decompose activations into J-space using gradient pursuit.

    The decomposition finds a sparse non-negative combination of J-lens vectors
    that best approximates the activation:
        min ‖h - Σ_i α_i v_{t_i}‖² s.t. α_i ≥ 0, |{i : α_i > 0}| ≤ k

    Uses gradient pursuit: iteratively select the atom with the largest positive
    inner product with the residual, then solve NNLS on the active set.
    """

    def __init__(
        self,
        jlens_vectors: torch.Tensor,
        k: int = 25,
        device: torch.device | str = "cpu",
    ):
        """Initialize the decomposer.

        Args:
            jlens_vectors: J-lens vectors of shape (vocab_size, d_model).
                          These are the rows of W_U · J_ℓ.
            k: Maximum sparsity level (number of active atoms).
            device: Computation device.
        """
        self.k = k
        self.device = torch.device(device)

        # Normalize J-lens vectors for efficient inner products
        self.vectors = jlens_vectors.to(self.device, dtype=torch.float32)
        self.norms = self.vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.vectors_normalized = self.vectors / self.norms

        self.vocab_size, self.d_model = self.vectors.shape

    def decompose(
        self,
        h: torch.Tensor,
        k: int | None = None,
        exclude_tokens: set[int] | None = None,
    ) -> JSpaceResult:
        """Decompose an activation into J-space via gradient pursuit.

        Args:
            h: Activation vector of shape (d_model,) or (1, d_model).
            k: Override sparsity level for this decomposition.
            exclude_tokens: Token indices to exclude from selection
                           (e.g., top-k clean-pass tokens for ablation).

        Returns:
            JSpaceResult with the decomposition.
        """
        if k is None:
            k = self.k

        h_flat = h.flatten().to(self.device, dtype=torch.float32)
        h_norm_sq = (h_flat ** 2).sum().item()

        if h_norm_sq < 1e-12:
            return JSpaceResult(
                coefficients=torch.zeros(0),
                indices=torch.zeros(0, dtype=torch.long),
                jspace_component=torch.zeros_like(h_flat),
                residual=h_flat.clone(),
                variance_explained=0.0,
                top_tokens=[],
            )

        # Gradient pursuit with non-negativity
        active_indices: list[int] = []
        residual = h_flat.clone()

        for _ in range(k):
            # Compute inner products with residual
            scores = torch.matmul(self.vectors, residual)  # (vocab_size,)

            # Enforce non-negativity: only consider positive projections
            scores = scores.clamp(min=0)

            # Exclude already-selected and forbidden tokens
            if active_indices:
                scores[torch.tensor(active_indices, device=self.device)] = 0
            if exclude_tokens:
                for idx in exclude_tokens:
                    if idx < self.vocab_size:
                        scores[idx] = 0

            # Select best atom
            best_idx = scores.argmax().item()
            if scores[best_idx] < 1e-8:
                break  # No more positive projections

            active_indices.append(best_idx)

            # Solve NNLS on active set
            V_active = self.vectors[active_indices]  # (n_active, d_model)
            coeffs = self._solve_nnls(V_active, h_flat)

            # Remove atoms with zero coefficients
            nonzero_mask = coeffs > 1e-8
            if not nonzero_mask.all():
                active_indices = [
                    idx for idx, nz in zip(active_indices, nonzero_mask.tolist()) if nz
                ]
                coeffs = coeffs[nonzero_mask]
                V_active = self.vectors[active_indices]

            # Update residual
            jspace_component = (coeffs.unsqueeze(1) * V_active).sum(dim=0)
            residual = h_flat - jspace_component

        # Compute final result
        if active_indices:
            V_active = self.vectors[active_indices]
            coeffs = self._solve_nnls(V_active, h_flat)
            jspace_component = (coeffs.unsqueeze(1) * V_active).sum(dim=0)
            residual = h_flat - jspace_component
        else:
            coeffs = torch.zeros(0, device=self.device)
            jspace_component = torch.zeros_like(h_flat)
            residual = h_flat.clone()

        # Variance explained
        jspace_var = (jspace_component ** 2).sum().item()
        variance_explained = jspace_var / h_norm_sq if h_norm_sq > 0 else 0.0

        # Build top tokens list
        indices_tensor = torch.tensor(active_indices, dtype=torch.long, device=self.device)
        top_tokens = [
            (idx, float(coeff))
            for idx, coeff in sorted(
                zip(active_indices, coeffs.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        return JSpaceResult(
            coefficients=coeffs.cpu(),
            indices=indices_tensor.cpu(),
            jspace_component=jspace_component.cpu(),
            residual=residual.cpu(),
            variance_explained=variance_explained,
            top_tokens=top_tokens,
        )

    def batch_decompose(
        self,
        activations: torch.Tensor,
        k: int | None = None,
        exclude_tokens: set[int] | None = None,
    ) -> list[JSpaceResult]:
        """Decompose a batch of activations.

        Args:
            activations: Tensor of shape (batch, d_model) or (batch, seq_len, d_model).
            k: Override sparsity level.
            exclude_tokens: Tokens to exclude.

        Returns:
            List of JSpaceResult, one per activation vector.
        """
        if activations.dim() == 3:
            batch, seq_len, d = activations.shape
            flat = activations.reshape(-1, d)
        else:
            flat = activations

        results = []
        for i in range(flat.shape[0]):
            results.append(self.decompose(flat[i], k=k, exclude_tokens=exclude_tokens))
        return results

    def _solve_nnls(self, V: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Solve non-negative least squares: min ‖h - V^T α‖² s.t. α ≥ 0.

        Uses the active-set method for small k (≤50).

        Args:
            V: Active vectors, shape (n_active, d_model).
            h: Target vector, shape (d_model,).

        Returns:
            Coefficients, shape (n_active,).
        """
        n_active = V.shape[0]

        if n_active == 0:
            return torch.zeros(0, device=self.device)

        if n_active == 1:
            # Simple projection
            coeff = torch.dot(V[0], h) / (torch.dot(V[0], V[0]) + 1e-8)
            return torch.clamp(coeff.unsqueeze(0), min=0)

        # Gram matrix: G = V V^T
        G = torch.matmul(V, V.T)  # (n_active, n_active)
        # Right-hand side: b = V h
        b = torch.matmul(V, h)  # (n_active,)

        # Solve unconstrained first
        try:
            L = torch.linalg.cholesky(G + 1e-6 * torch.eye(n_active, device=self.device))
            alpha = torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
        except torch.linalg.LinAlgError:
            # Fallback to pseudoinverse
            alpha = torch.linalg.lstsq(G, b).solution

        # Project to non-negative
        alpha = torch.clamp(alpha, min=0)

        # Iterative correction (simplified active-set)
        for _ in range(10):
            mask = alpha > 1e-8
            if mask.all() or not mask.any():
                break
            # Re-solve on active subset
            active_mask = mask
            G_sub = G[active_mask][:, active_mask]
            b_sub = b[active_mask]
            try:
                n_sub = G_sub.shape[0]
                L_sub = torch.linalg.cholesky(
                    G_sub + 1e-6 * torch.eye(n_sub, device=self.device)
                )
                alpha_sub = torch.cholesky_solve(b_sub.unsqueeze(1), L_sub).squeeze(1)
            except torch.linalg.LinAlgError:
                break
            alpha = torch.zeros_like(alpha)
            alpha[active_mask] = torch.clamp(alpha_sub, min=0)

        return alpha

    def compute_occupancy(
        self,
        activations: torch.Tensor,
        max_k: int = 50,
        threshold: float = 0.01,
    ) -> list[int]:
        """Compute the effective occupancy (number of active J-space atoms) per position.

        Occupancy is the k at which reconstruction gain over random-direction controls
        vanishes. Simplified here as the number of atoms with coefficient > threshold.

        Args:
            activations: Tensor of shape (seq_len, d_model) or (batch, seq_len, d_model).
            max_k: Maximum k to test.
            threshold: Minimum coefficient to count as active.

        Returns:
            List of occupancy values per position.
        """
        if activations.dim() == 3:
            activations = activations[0]  # Take first batch element

        occupancies = []
        for pos in range(activations.shape[0]):
            result = self.decompose(activations[pos], k=max_k)
            n_active = (result.coefficients > threshold).sum().item()
            occupancies.append(int(n_active))

        return occupancies
