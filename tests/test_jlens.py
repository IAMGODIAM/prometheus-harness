"""Tests for J-Lens core modules."""

import pytest
import torch
import torch.nn as nn

from prometheus.jlens import (
    JacobianLens,
    LensConfig,
    JSpaceDecomposer,
    JSpaceResult,
    SteeringIntervention,
    AblationIntervention,
    ConceptSwapIntervention,
    InterventionHook,
    Watchlist,
    WatchlistScores,
)


class TestJacobianLens:
    """Tests for the JacobianLens class."""

    def _make_lens(self, d_model=64, n_layers=12, source_layers=None):
        """Helper to create a test lens."""
        if source_layers is None:
            source_layers = list(range(4, 8))
        config = LensConfig(
            model_id="test-model",
            d_model=d_model,
            n_layers=n_layers,
            source_layers=source_layers,
            target_layer=n_layers - 1,
        )
        matrices = {layer: torch.randn(d_model, d_model) for layer in source_layers}
        return JacobianLens(config=config, matrices=matrices)

    def test_lens_creation(self):
        """Test creating a J-lens with given dimensions."""
        lens = self._make_lens(d_model=768, n_layers=12)
        assert lens.config.d_model == 768
        assert lens.config.n_layers == 12

    def test_lens_apply_shape(self):
        """Test that lens application produces correct output shape."""
        d_model = 64
        vocab_size = 100
        lens = self._make_lens(d_model=d_model, n_layers=4, source_layers=[0, 1, 2, 3])

        h = torch.randn(1, d_model)
        unembed = torch.randn(vocab_size, d_model)

        logits = lens.apply(h, layer=2, norm_fn=None, unembed=unembed)
        assert logits.shape == (1, vocab_size)

    def test_lens_apply_without_unembed(self):
        """Test that lens without unembed returns transformed hidden state."""
        d_model = 64
        lens = self._make_lens(d_model=d_model, n_layers=4, source_layers=[0, 1, 2, 3])

        h = torch.randn(1, d_model)
        transformed = lens.apply(h, layer=2, norm_fn=None, unembed=None)
        assert transformed.shape == (1, d_model)

    def test_lens_apply_invalid_layer(self):
        """Test that applying at an unavailable layer raises ValueError."""
        lens = self._make_lens(d_model=64, n_layers=12, source_layers=[4, 5, 6])
        h = torch.randn(1, 64)
        with pytest.raises(ValueError, match="Layer 10 not in lens"):
            lens.apply(h, layer=10)

    def test_lens_save_load(self, tmp_path):
        """Test lens serialization and deserialization."""
        d_model = 64
        lens = self._make_lens(d_model=d_model, n_layers=12, source_layers=[4, 5, 6, 7])

        path = tmp_path / "test_lens"
        lens.save(str(path))

        loaded = JacobianLens.load(str(path))
        assert loaded.config.d_model == d_model
        assert loaded.config.n_layers == 12
        assert loaded.config.model_id == "test-model"
        assert set(loaded.matrices.keys()) == {4, 5, 6, 7}

        # Check matrices are approximately equal
        for layer in [4, 5, 6, 7]:
            assert torch.allclose(lens.matrices[layer], loaded.matrices[layer])

    def test_lens_top_k_tokens(self):
        """Test top-k token retrieval."""
        d_model = 64
        vocab_size = 100
        lens = self._make_lens(d_model=d_model, n_layers=4, source_layers=[0, 1, 2, 3])

        h = torch.randn(1, d_model)
        unembed = torch.randn(vocab_size, d_model)

        values, indices = lens.top_k_tokens(h, layer=2, unembed=unembed, k=5)
        assert values.shape == (1, 5)
        assert indices.shape == (1, 5)

    def test_lens_merge(self):
        """Test merging multiple lenses."""
        lens1 = self._make_lens(d_model=64, n_layers=4, source_layers=[0, 1, 2, 3])
        lens2 = self._make_lens(d_model=64, n_layers=4, source_layers=[0, 1, 2, 3])

        merged = JacobianLens.merge([lens1, lens2])
        assert set(merged.matrices.keys()) == {0, 1, 2, 3}
        # Merged should be average of the two
        for layer in [0, 1, 2, 3]:
            expected = (lens1.matrices[layer] + lens2.matrices[layer]) / 2
            assert torch.allclose(merged.matrices[layer], expected, atol=1e-6)

    def test_lens_id_deterministic(self):
        """Test that lens_id is deterministic for same config."""
        lens1 = self._make_lens()
        lens2 = self._make_lens()
        assert lens1.lens_id == lens2.lens_id


class TestJSpaceDecomposer:
    """Tests for J-Space sparse decomposition."""

    def test_decompose_basic(self):
        """Test basic decomposition produces valid output."""
        d_model = 64
        vocab_size = 100
        k = 10

        jlens_vectors = torch.randn(vocab_size, d_model)
        decomposer = JSpaceDecomposer(jlens_vectors, k=k)

        h = torch.randn(d_model)
        result = decomposer.decompose(h)

        assert isinstance(result, JSpaceResult)
        assert len(result.top_tokens) <= k
        assert 0.0 <= result.variance_explained <= 1.0

    def test_decompose_sparse(self):
        """Test that decomposition is actually sparse."""
        d_model = 32
        vocab_size = 50
        k = 5

        jlens_vectors = torch.randn(vocab_size, d_model)
        decomposer = JSpaceDecomposer(jlens_vectors, k=k)

        h = torch.randn(d_model)
        result = decomposer.decompose(h)

        assert len(result.top_tokens) <= k

    def test_decompose_nonnegative_coefficients(self):
        """Test that coefficients are non-negative."""
        d_model = 32
        vocab_size = 50
        k = 10

        jlens_vectors = torch.randn(vocab_size, d_model)
        decomposer = JSpaceDecomposer(jlens_vectors, k=k)

        h = torch.randn(d_model)
        result = decomposer.decompose(h)

        for _, coeff in result.top_tokens:
            assert coeff >= 0.0

    def test_decompose_reconstruction(self):
        """Test that jspace_component + residual = original."""
        d_model = 32
        vocab_size = 50
        k = 10

        jlens_vectors = torch.randn(vocab_size, d_model)
        decomposer = JSpaceDecomposer(jlens_vectors, k=k)

        h = torch.randn(d_model)
        result = decomposer.decompose(h)

        reconstructed = result.jspace_component + result.residual
        assert torch.allclose(h, reconstructed, atol=1e-5)

    def test_decompose_zero_vector(self):
        """Test decomposition of zero vector."""
        d_model = 32
        vocab_size = 50

        jlens_vectors = torch.randn(vocab_size, d_model)
        decomposer = JSpaceDecomposer(jlens_vectors, k=10)

        h = torch.zeros(d_model)
        result = decomposer.decompose(h)

        assert result.variance_explained == 0.0
        assert len(result.top_tokens) == 0

    def test_batch_decompose(self):
        """Test batch decomposition."""
        d_model = 32
        vocab_size = 50
        batch_size = 4

        jlens_vectors = torch.randn(vocab_size, d_model)
        decomposer = JSpaceDecomposer(jlens_vectors, k=10)

        activations = torch.randn(batch_size, d_model)
        results = decomposer.batch_decompose(activations)

        assert len(results) == batch_size
        for r in results:
            assert isinstance(r, JSpaceResult)


class TestInterventions:
    """Tests for J-lens interventions."""

    def test_steering_intervention(self):
        """Test steering intervention modifies activations."""
        d_model = 64
        direction = torch.randn(d_model)
        direction = direction / direction.norm()

        intervention = SteeringIntervention(
            layers=[2, 3, 4],
            direction=direction,
            alpha=2.0,
        )

        h = torch.randn(10, d_model)  # (seq_len, d_model)
        h_original = h.clone()
        h_steered = intervention.apply(h.clone(), layer=3, position=5)

        # Should be different from original at position 5
        assert not torch.allclose(h_original[5], h_steered[5])
        # Other positions should be unchanged
        assert torch.allclose(h_original[0], h_steered[0])

    def test_steering_wrong_layer(self):
        """Test that steering doesn't apply at wrong layer."""
        d_model = 64
        direction = torch.randn(d_model)

        intervention = SteeringIntervention(
            layers=[5, 6],
            direction=direction,
            alpha=1.0,
        )

        h = torch.randn(10, d_model)
        h_original = h.clone()
        h_result = intervention.apply(h.clone(), layer=3, position=0)
        assert torch.allclose(h_original, h_result)

    def test_ablation_intervention(self):
        """Test ablation removes specified direction."""
        d_model = 64
        direction = torch.randn(d_model)
        direction = direction / direction.norm()

        intervention = AblationIntervention(
            layers=[2, 3],
            directions=[direction],
        )

        # Create activation with known component along direction
        h = direction.unsqueeze(0).expand(10, -1) * 5.0  # (10, d_model)
        h_ablated = intervention.apply(h.clone(), layer=2, position=0)

        # Component along direction should be near zero at position 0
        projection = torch.dot(h_ablated[0], direction)
        assert abs(projection.item()) < 0.01

    def test_concept_swap(self):
        """Test concept swap modifies activations."""
        d_model = 64
        v_s = torch.randn(d_model)
        v_t = torch.randn(d_model)

        intervention = ConceptSwapIntervention(
            layers=[3],
            source_direction=v_s,
            target_direction=v_t,
            alpha=1.0,
        )

        h = torch.randn(10, d_model)
        h_swapped = intervention.apply(h.clone(), layer=3, position=5)

        # Should be different at position 5
        assert not torch.allclose(h[5], h_swapped[5])

    def test_intervention_hook_context_manager(self):
        """Test InterventionHook as context manager."""
        direction = torch.randn(64)
        intervention = SteeringIntervention(
            layers=[0],
            direction=direction,
            alpha=1.0,
        )

        hook = InterventionHook([intervention])
        assert len(hook._hooks) == 0
        # Context manager should clean up
        with hook:
            pass
        assert len(hook._hooks) == 0


class TestWatchlist:
    """Tests for the watchlist module."""

    def test_watchlist_creation(self):
        """Test creating a watchlist with default categories."""
        wl = Watchlist()
        assert "deception" in wl.categories
        assert "prompt_injection" in wl.categories
        assert len(wl.categories) >= 5

    def test_add_category(self):
        """Test adding a custom category."""
        wl = Watchlist(categories={}, thresholds={})
        wl.add_category("test_cat", ["hello", "world"], threshold=0.2)
        assert "test_cat" in wl.categories
        assert wl.thresholds["test_cat"] == 0.2

    def test_remove_category(self):
        """Test removing a category."""
        wl = Watchlist()
        wl.remove_category("deception")
        assert "deception" not in wl.categories

    def test_to_from_dict(self):
        """Test serialization round-trip."""
        wl = Watchlist()
        wl.add_category("custom", ["test"], threshold=0.3)

        data = wl.to_dict()
        wl2 = Watchlist.from_dict(data)

        assert wl2.categories == wl.categories
        assert wl2.thresholds == wl.thresholds

    def test_default_thresholds(self):
        """Test that default thresholds are set."""
        wl = Watchlist()
        for category in wl.categories:
            assert category in wl.thresholds
            assert wl.thresholds[category] > 0


class TestLensConfig:
    """Tests for LensConfig."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = LensConfig(
            model_id="gpt2",
            d_model=768,
            n_layers=12,
            source_layers=[4, 5, 6, 7],
            target_layer=11,
        )
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.target_layer == 11

    def test_config_lens_id_deterministic(self):
        """Test that lens_id is deterministic."""
        config1 = LensConfig(
            model_id="gpt2",
            d_model=768,
            n_layers=12,
            source_layers=[4, 5, 6, 7],
            target_layer=11,
        )
        config2 = LensConfig(
            model_id="gpt2",
            d_model=768,
            n_layers=12,
            source_layers=[4, 5, 6, 7],
            target_layer=11,
        )
        assert config1.lens_id() == config2.lens_id()

    def test_config_lens_id_changes_with_params(self):
        """Test that lens_id changes with different params."""
        config1 = LensConfig(
            model_id="gpt2",
            d_model=768,
            n_layers=12,
            source_layers=[4, 5, 6, 7],
            target_layer=11,
        )
        config2 = LensConfig(
            model_id="llama",
            d_model=768,
            n_layers=12,
            source_layers=[4, 5, 6, 7],
            target_layer=11,
        )
        assert config1.lens_id() != config2.lens_id()
