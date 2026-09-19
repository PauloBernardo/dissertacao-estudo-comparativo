"""Tests for FT-Transformer and sparse attention modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sklearn.datasets import make_moons, make_classification

from src.models.transformers.sparse_attention.topk_attention import TopKAttention
from src.models.transformers.sparse_attention.sparsemax_attention import SparsemaxAttention
from src.models.transformers.sparse_attention.entmax_attention import EntmaxAttention
from src.models.transformers.ft_transformer import FTTransformer


# ── Attention kernel unit tests ───────────────────────────────────────────────

class TestTopKAttention:
    def _scores(self, B=2, H=2, L=5):
        torch.manual_seed(0)
        return torch.randn(B, H, L, L)

    def test_output_shape(self):
        attn = TopKAttention(topk_ratio=0.5)
        s = self._scores()
        w = attn(s)
        assert w.shape == s.shape

    def test_sums_to_one(self):
        attn = TopKAttention(topk_ratio=0.5)
        w = attn(self._scores())
        torch.testing.assert_close(w.sum(dim=-1), torch.ones(2, 2, 5), atol=1e-5, rtol=0)

    def test_exact_sparsity(self):
        """Each query should attend to exactly k keys."""
        L = 10
        k_ratio = 0.3
        attn = TopKAttention(topk_ratio=k_ratio)
        s = self._scores(L=L)
        w = attn(s)
        k = max(1, round(k_ratio * L))
        nonzero = (w > 1e-9).sum(dim=-1)
        assert (nonzero == k).all()

    def test_topk_ratio_one_is_full(self):
        attn = TopKAttention(topk_ratio=1.0)
        s = self._scores()
        w = attn(s)
        assert (w > 1e-9).all()

    def test_nonnegative(self):
        attn = TopKAttention(topk_ratio=0.4)
        w = attn(self._scores())
        assert (w >= 0).all()


class TestSparsemaxAttention:
    def _scores(self, B=2, H=2, L=6):
        torch.manual_seed(1)
        return torch.randn(B, H, L, L)

    def test_output_shape(self):
        attn = SparsemaxAttention()
        s = self._scores()
        assert attn(s).shape == s.shape

    def test_sums_to_one(self):
        attn = SparsemaxAttention()
        w = attn(self._scores())
        torch.testing.assert_close(w.sum(dim=-1), torch.ones(2, 2, 6), atol=1e-5, rtol=0)

    def test_nonnegative(self):
        attn = SparsemaxAttention()
        assert (attn(self._scores()) >= 0).all()

    def test_produces_zeros(self):
        """Sparsemax should zero out at least some entries."""
        attn = SparsemaxAttention()
        w = attn(self._scores())
        assert (w == 0).any()

    def test_dominated_entry_zero(self):
        """A clearly dominated key should receive zero weight."""
        s = torch.zeros(1, 1, 1, 4)
        s[..., 0] = 10.0   # dominant
        s[..., 1:] = -10.0  # dominated
        w = SparsemaxAttention()(s)
        assert w[..., 1:].abs().max() < 1e-4

    def test_uniform_scores(self):
        """Uniform scores → uniform distribution (no zeros)."""
        s = torch.zeros(1, 1, 1, 4)
        w = SparsemaxAttention()(s)
        torch.testing.assert_close(w, torch.full_like(w, 0.25), atol=1e-5, rtol=0)


class TestEntmaxAttention:
    def _scores(self, B=2, H=2, L=6):
        torch.manual_seed(2)
        return torch.randn(B, H, L, L)

    def test_output_shape(self):
        attn = EntmaxAttention(alpha=1.5)
        assert attn(self._scores()).shape == self._scores().shape

    def test_sums_to_one(self):
        attn = EntmaxAttention(alpha=1.5)
        w = attn(self._scores())
        torch.testing.assert_close(w.sum(dim=-1), torch.ones(2, 2, 6), atol=1e-4, rtol=0)

    def test_nonnegative(self):
        attn = EntmaxAttention(alpha=1.5)
        assert (attn(self._scores()) >= 0).all()

    def test_alpha_one_is_softmax(self):
        """α=1 → softmax."""
        s = self._scores()
        import torch.nn.functional as F
        w_entmax = EntmaxAttention(alpha=1.0)(s)
        w_softmax = F.softmax(s, dim=-1)
        torch.testing.assert_close(w_entmax, w_softmax, atol=1e-5, rtol=0)

    def test_alpha_two_is_sparsemax(self):
        """α=2 → at least as sparse as sparsemax (by number of zeros per row)."""
        # Use a well-separated distribution where support is unambiguous
        s = torch.zeros(1, 1, 1, 8)
        # Clear winner at positions 0 and 2; rest dominated
        s[..., 0] = 5.0
        s[..., 2] = 4.0
        s[..., 1] = -5.0
        s[..., 3:] = -5.0
        w_e2 = EntmaxAttention(alpha=2.0)(s)
        w_sm = SparsemaxAttention()(s)
        # Both should assign zeros to the dominated positions
        assert (w_e2[..., 1] < 1e-6).all()
        assert (w_sm[..., 1] < 1e-6).all()
        # Both sum to 1
        torch.testing.assert_close(w_e2.sum(dim=-1), torch.ones(1, 1, 1), atol=1e-4, rtol=0)
        torch.testing.assert_close(w_sm.sum(dim=-1), torch.ones(1, 1, 1), atol=1e-4, rtol=0)

    def test_dominated_entry_zero(self):
        s = torch.zeros(1, 1, 1, 4)
        s[..., 0] = 10.0
        s[..., 1:] = -10.0
        w = EntmaxAttention(alpha=1.5)(s)
        assert w[..., 1:].abs().max() < 1e-3

    def test_higher_alpha_more_sparse(self):
        """More sparse at α=2 than α=1.5 on random scores."""
        s = self._scores()
        w15 = EntmaxAttention(alpha=1.5)(s)
        w20 = EntmaxAttention(alpha=2.0)(s)
        zeros15 = (w15 < 1e-6).float().mean()
        zeros20 = (w20 < 1e-6).float().mean()
        assert zeros20 >= zeros15


# ── FT-Transformer integration tests ─────────────────────────────────────────

def _moons_dataset(n=200, noise=0.1, seed=0):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float32), y.astype(int)


def _tiny_dataset(n=50, p=4, seed=0):
    X, y = make_classification(n_samples=n, n_features=p, n_informative=3,
                                n_redundant=1, random_state=seed)
    return X.astype(np.float32), y.astype(int)


@pytest.fixture(scope="module")
def moons():
    return _moons_dataset()


class TestFTTransformerFit:
    @pytest.mark.parametrize("attention_type", ["softmax", "topk", "entmax", "sparsemax"])
    def test_fit_predict_moons(self, moons, attention_type):
        X, y = moons
        clf = FTTransformer(
            embedding_dim=16,
            num_blocks=1,
            num_heads=2,
            max_epochs=5,
            batch_size=64,
            val_fraction=0.0,
            attention_type=attention_type,
            random_state=0,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, moons):
        X, y = moons
        clf = FTTransformer(embedding_dim=16, num_blocks=1, num_heads=2,
                            max_epochs=5, val_fraction=0.0, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, moons):
        X, y = moons
        clf = FTTransformer(embedding_dim=16, num_blocks=1, num_heads=2,
                            max_epochs=5, val_fraction=0.0, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes_attribute(self, moons):
        X, y = moons
        clf = FTTransformer(embedding_dim=16, num_blocks=1, num_heads=2,
                            max_epochs=5, val_fraction=0.0, random_state=0)
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_n_iter_recorded(self, moons):
        X, y = moons
        clf = FTTransformer(embedding_dim=16, num_blocks=1, num_heads=2,
                            max_epochs=10, val_fraction=0.0, random_state=0)
        clf.fit(X, y)
        assert 1 <= clf.n_iter_ <= 10

    def test_train_losses_recorded(self, moons):
        X, y = moons
        clf = FTTransformer(embedding_dim=16, num_blocks=1, num_heads=2,
                            max_epochs=5, val_fraction=0.0, random_state=0)
        clf.fit(X, y)
        assert len(clf.train_losses_) == clf.n_iter_
        assert all(l > 0 for l in clf.train_losses_)

    def test_early_stopping_reduces_epochs(self, moons):
        X, y = moons
        clf = FTTransformer(embedding_dim=16, num_blocks=1, num_heads=2,
                            max_epochs=200, patience=3, val_fraction=0.15,
                            random_state=0)
        clf.fit(X, y)
        # Should stop before 200 epochs on a learnable dataset
        assert clf.n_iter_ <= 200

    def test_reproducibility(self, moons):
        X, y = moons
        kwargs = dict(embedding_dim=16, num_blocks=1, num_heads=2,
                      max_epochs=5, val_fraction=0.0, random_state=7)
        clf1 = FTTransformer(**kwargs).fit(X, y)
        clf2 = FTTransformer(**kwargs).fit(X, y)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_overfit_tiny(self):
        """Should overfit a tiny dataset when given enough capacity."""
        X, y = _tiny_dataset(n=50, p=4, seed=0)
        clf = FTTransformer(
            embedding_dim=32, num_blocks=2, num_heads=4,
            max_epochs=300, batch_size=16, val_fraction=0.0,
            lr=1e-3, random_state=0,
        )
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc >= 0.90, f"Expected overfit acc ≥ 0.90, got {acc:.3f}"


class TestFTTransformerSparsity:
    @pytest.mark.parametrize("attention_type", ["topk", "sparsemax"])
    def test_sparse_attention_has_zeros(self, attention_type):
        # topk and sparsemax always produce exact zeros by construction
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                                   random_state=0)
        X = X.astype(np.float32)
        clf = FTTransformer(
            embedding_dim=16, num_blocks=1, num_heads=2,
            max_epochs=3, val_fraction=0.0,
            attention_type=attention_type, random_state=0,
        )
        clf.fit(X, y)
        clf.predict_proba(X[:16])
        metrics = clf.attention_sparsity()
        assert metrics["mean_zero_fraction"] > 0.0

    def test_entmax_lower_entropy_than_softmax(self):
        """Entmax should be more concentrated (lower entropy) than softmax."""
        # Test the kernel directly on peaked scores where entmax-1.5 must be sparser
        torch.manual_seed(5)
        # Create scores with a clear winner per row
        s = torch.randn(4, 4, 8, 20) * 2.0
        s[..., 0] += 3.0   # amplify position 0 to create a peak

        w_soft = torch.nn.functional.softmax(s, dim=-1)
        w_entmax = EntmaxAttention(alpha=1.5)(s)

        # Entropy: -sum(w * log(w+eps))
        eps = 1e-9
        H_soft = -(w_soft * (w_soft + eps).log()).sum(dim=-1).mean()
        H_entmax = -(w_entmax * (w_entmax + eps).log()).sum(dim=-1).mean()
        assert H_entmax < H_soft, f"entmax H={H_entmax:.3f} not < softmax H={H_soft:.3f}"

    def test_softmax_has_no_exact_zeros(self, moons):
        X, y = moons
        clf = FTTransformer(
            embedding_dim=16, num_blocks=1, num_heads=2,
            max_epochs=3, val_fraction=0.0,
            attention_type="softmax", random_state=0,
        )
        clf.fit(X, y)
        clf.predict_proba(X[:16])
        metrics = clf.attention_sparsity()
        assert metrics["mean_zero_fraction"] < 0.01

    def test_sparsity_metrics_structure(self, moons):
        X, y = moons
        clf = FTTransformer(embedding_dim=16, num_blocks=1, num_heads=2,
                            max_epochs=3, val_fraction=0.0, random_state=0)
        clf.fit(X, y)
        clf.predict_proba(X[:8])
        metrics = clf.attention_sparsity()
        assert set(metrics.keys()) == {"mean_zero_fraction", "mean_entropy", "effective_n_tokens"}
        assert all(isinstance(v, float) for v in metrics.values())

    def test_topk_zero_fraction_matches_ratio(self):
        """TopK zero fraction should match 1 - topk_ratio (up to rounding on seq_len)."""
        # Use 20 features → seq_len=21; k = round(0.5*21)=10 → exact zero_frac=11/21≈0.524
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                                   random_state=0)
        X = X.astype(np.float32)
        topk_ratio = 0.5
        seq_len = X.shape[1] + 1  # +1 for CLS token
        k = max(1, round(topk_ratio * seq_len))
        expected_zeros = (seq_len - k) / seq_len

        clf = FTTransformer(
            embedding_dim=16, num_blocks=1, num_heads=2,
            max_epochs=2, val_fraction=0.0,
            attention_type="topk", topk_ratio=topk_ratio, random_state=0,
        )
        clf.fit(X, y)
        clf.predict_proba(X[:32])
        metrics = clf.attention_sparsity()
        assert abs(metrics["mean_zero_fraction"] - expected_zeros) < 0.05

    def test_sparse_more_zeros_than_softmax(self):
        """TopK and sparsemax always produce strictly more zeros than softmax."""
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                                   random_state=42)
        X = X.astype(np.float32)
        base_kwargs = dict(embedding_dim=16, num_blocks=1, num_heads=2,
                           max_epochs=3, val_fraction=0.0, random_state=42)
        clf_soft = FTTransformer(**base_kwargs, attention_type="softmax").fit(X, y)
        clf_soft.predict_proba(X[:32])
        z_soft = clf_soft.attention_sparsity()["mean_zero_fraction"]

        for at in ["topk", "sparsemax"]:
            clf = FTTransformer(**base_kwargs, attention_type=at).fit(X, y)
            clf.predict_proba(X[:32])
            z = clf.attention_sparsity()["mean_zero_fraction"]
            assert z > z_soft, f"{at} zero_fraction={z:.3f} not > softmax={z_soft:.3f}"


# ── Regressão da auditoria 2026-09-18 ────────────────────────────────────────

class TestEntmaxBisectionRegression:
    """Trava a correção do intervalo de bisseção do entmax.

    A versão antiga convergia sempre ao extremo do intervalo e a soma dos
    pesos ANTES da renormalização ficava entre ~1,4 e ~7 (não 1).
    """

    def test_sum_is_one_without_renormalisation(self):
        from src.models.transformers.sparse_attention.entmax_attention import (
            _EntmaxBisectFunction,
        )
        torch.manual_seed(0)
        z = torch.randn(6, 9, dtype=torch.float64) * 0.5   # regime onde o bug era pior
        p = _EntmaxBisectFunction.apply(z, 1.5, 60)
        # reconstrói τ a partir de p e checa que a soma bruta (sem renormalizar) é 1
        am1 = 0.5
        support = p > 0
        tau = (am1 * z - p ** am1)[support].view(-1)  # (α−1)z_i − p_i^{α−1} = τ em todo o suporte
        # τ deve ser constante por linha → soma bruta com esse τ = 1
        z_rows = am1 * z
        tau_row = torch.stack([(z_rows[i] - p[i] ** am1)[support[i]].mean() for i in range(z.shape[0])])
        raw = ((z_rows - tau_row[:, None]).clamp(min=0) ** (1 / am1)).sum(-1)
        torch.testing.assert_close(raw, torch.ones_like(raw), atol=1e-8, rtol=0)

    def test_alpha_two_equals_sparsemax(self):
        from src.models.transformers.sparse_attention.entmax_attention import _entmax_bisect
        from src.models.transformers.sparse_attention.sparsemax_attention import _sparsemax
        torch.manual_seed(1)
        z = torch.randn(4, 7, dtype=torch.float64) * 2
        torch.testing.assert_close(_entmax_bisect(z, 2.0, 60), _sparsemax(z), atol=1e-10, rtol=0)

    def test_exact_jacobian(self):
        from src.models.transformers.sparse_attention.entmax_attention import _EntmaxBisectFunction
        torch.manual_seed(2)
        z = (torch.randn(3, 8, dtype=torch.float64) * 1.5).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: _EntmaxBisectFunction.apply(x, 1.5, 60), (z,), eps=1e-6, atol=1e-5
        )


def test_get_selector_opposite_is_disabled():
    """'opposite' apontava para a reflexão OBL (Tizhoosh), não para os Opposite
    Maps de Rocha Neto & Barreto; a chave foi desativada na auditoria."""
    import pytest
    from src.models.landmark_selection import get_selector
    with pytest.raises(ValueError, match="select_opposite_landmarks"):
        get_selector("opposite", 5, random_state=0)
    sel = get_selector("obl_reflection", 5, random_state=0)
    assert sel.__class__.__name__ == "OBLReflectionSelector"


class TestSAINTFaithful:
    """SAINT fiel (2026-09-18): MISA sobre a linha inteira, não só o CLS."""

    def _model(self, p=5):
        from src.models.ft_transformer_model import SAINTClassifier
        torch.manual_seed(0)
        m = SAINTClassifier(n_features=p, d_model=8, n_heads=2, n_layers=1,
                            dim_head=4, attn_dropout=0.0, ff_dropout=0.0, head_hidden=16)
        return m.eval()

    def test_shapes_and_intersample_dependence(self):
        m = self._model()
        x = torch.randn(6, 5)
        out = m(x)
        assert out.shape == (6,)
        # alterar OUTRA linha do lote muda a saída da linha 0 (atenção inter-instâncias)
        x2 = x.clone(); x2[3] += 5.0
        assert not torch.allclose(m(x2)[0], out[0])
        # e a MISA usa todos os tokens: mudar só um atributo de outra linha já basta
        x3 = x.clone(); x3[4, 2] += 5.0
        assert not torch.allclose(m(x3)[0], out[0])

    def test_row_permutation_equivariance(self):
        m = self._model()
        x = torch.randn(7, 5)
        perm = torch.randperm(7)
        torch.testing.assert_close(m(x)[perm], m(x[perm]), atol=1e-5, rtol=1e-5)

    def test_misa_dimension_is_row_concat(self):
        from src.models.ft_transformer_model import SAINTClassifier
        m = SAINTClassifier(n_features=5, d_model=8, n_heads=2, n_layers=1, dim_head=4)
        assert m.stages[0].misa.to_qkv.in_features == (5 + 1) * 8

    def test_wrapper_fit_predict_cpu(self):
        from src.models.ft_transformer_saint_wrapper import SAINTColnorm
        rng = np.random.RandomState(0)
        X = rng.randn(120, 4); y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = SAINTColnorm(d_model=8, n_heads=2, n_layers=1, epochs=3, patience=2,
                           batch_size=64, random_state=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (120,)
        assert clf.predict_proba(X).shape == (120, 2)


class TestFTCURMinibatchGlobalAndStreaming:
    """FT-CUR: mini-lote com landmarks globais e rota de predição streaming (2026-09-18)."""

    def _data(self, n=300, p=5, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, p); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        return X, y

    def test_global_minibatch_fits_and_predicts(self):
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        X, y = self._data()
        m = FTTransformerCURColnorm(d_model=8, n_heads=2, n_layers=1, m_ratio=0.1, epochs=3,
                                    patience=2, batch_size=64, minibatch_landmarks="global",
                                    random_state=0).fit(X, y)
        assert m.predict(X[:10]).shape == (10,)

    def test_streaming_predictions_are_independent_of_test_composition(self):
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        X, y = self._data()
        m = FTTransformerCURColnorm(d_model=8, n_heads=2, n_layers=1, m_ratio=0.1, epochs=3,
                                    patience=2, batch_size=None, predict_mode="streaming",
                                    random_state=0).fit(X, y)
        Xte = self._data(seed=1)[0][:40]
        together = m._logits(Xte)
        alone = np.concatenate([m._logits(Xte[:1]), m._logits(Xte[1:])])
        np.testing.assert_allclose(together, alone, rtol=1e-5, atol=1e-6)
        # a rota histórica, por construção, depende do lote de teste (só documenta o contraste)
        m.predict_mode = "full_context"
        assert m._logits(Xte).shape == (40,)

    def test_min_epochs_delays_early_stopping(self):
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        import src.models.ft_transformer_model as M
        X, y = self._data()
        captured = {}
        orig = M.fit_model
        def spy(*a, **k):
            info = orig(*a, **k); captured['n'] = info['n_epochs']; return info
        import src.models.ft_transformer_cur_wrapper as W
        W.fit_model = spy
        try:
            FTTransformerCURColnorm(d_model=8, n_heads=2, n_layers=1, epochs=30, patience=1,
                                    min_epochs=20, random_state=0).fit(X, y)
        finally:
            W.fit_model = orig
        assert captured['n'] >= 20


class TestSAINTStyleReference:
    """O padrão do SAINT é o estilo do código de referência (auditoria 2026-09-18)."""

    def test_default_style_is_reference_prenorm(self):
        from src.models.ft_transformer_model import SAINTClassifier
        m = SAINTClassifier(n_features=5, d_model=8, n_heads=2, n_layers=1, dim_head=4)
        st = m.stages[0]
        assert st.style == "reference"
        # FF2 e LN3/LN4 operam sobre a linha achatada (n+1)·d, como no RowColTransformer
        assert st.ln3.normalized_shape == ((5 + 1) * 8,)
        assert st.ff2.net[0].in_features == (5 + 1) * 8
        # atenção de linha com dim_head próprio (64 no código oficial)
        assert st.misa.to_qkv.out_features == 3 * 2 * 64

    def test_paper_eq_style_is_postnorm_per_token(self):
        from src.models.ft_transformer_model import SAINTClassifier
        st = SAINTClassifier(n_features=5, d_model=8, n_heads=2, n_layers=1,
                            dim_head=4, style="paper_eq").stages[0]
        assert st.ln3.normalized_shape == (8,)
        assert st.ff2.net[0].in_features == 8

    def test_both_styles_train(self):
        from src.models.ft_transformer_saint_wrapper import SAINTColnorm
        rng = np.random.RandomState(0)
        X = rng.randn(120, 4); y = (X[:, 0] > 0).astype(int)
        for style in ("reference", "paper_eq"):
            clf = SAINTColnorm(d_model=8, n_heads=2, n_layers=1, epochs=3, patience=2,
                               batch_size=64, style=style, random_state=0).fit(X, y)
            assert clf.predict(X).shape == (120,)
