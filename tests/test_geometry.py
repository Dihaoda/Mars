import math

import pytest
import torch

from mars.geometry import Matrix, aggregate, combine, delta, effective, features, inner, norm


def state(seed, rank=3):
    generator = torch.Generator().manual_seed(seed)
    return {"layer.lora_A.weight": torch.randn(rank, 7, generator=generator, dtype=torch.float64),
            "layer.lora_B.weight": torch.randn(9, rank, generator=generator, dtype=torch.float64)}


def test_low_rank_inner_norm_match_dense():
    a, b = effective(state(1), 2), effective(state(2), 2)
    dense_a, dense_b = a["layer."].dense(), b["layer."].dense()
    assert inner(a, b) == pytest.approx(float((dense_a * dense_b).sum()), rel=1e-10)
    assert norm(a) == pytest.approx(float(dense_a.norm()), rel=1e-10)


def test_actual_delta_includes_cross_terms():
    local, base = state(1), state(2)
    actual = delta(local, base, 2)["layer."].dense()
    expected = effective(local, 2)["layer."].dense() - effective(base, 2)["layer."].dense()
    incorrect = 2 * (local["layer.lora_B.weight"] - base["layer.lora_B.weight"]) @ (local["layer.lora_A.weight"] - base["layer.lora_A.weight"])
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, incorrect)


def test_effective_features_invariant_under_gauge_change():
    local, base, root = state(1), state(2), state(3)
    transform = torch.diag(torch.tensor([2.0, 0.5, 3.0], dtype=torch.float64))
    changed = {"layer.lora_B.weight": local["layer.lora_B.weight"] @ transform,
               "layer.lora_A.weight": torch.linalg.solve(transform, local["layer.lora_A.weight"])}
    assert features(local, base, root, 2, "effective") == pytest.approx(features(changed, base, root, 2, "effective"), rel=1e-10)
    assert features(local, base, root, 2, "raw") != pytest.approx(features(changed, base, root, 2, "raw"))


def test_truncation_error_equals_discarded_singular_values():
    base, one, two = state(0), state(1), state(2)
    result, info = aggregate(base, [one, two], [0.25, 0.75], one, 2)
    target = 0.25 * effective(one, 2)["layer."].dense() + 0.75 * effective(two, 2)["layer."].dense()
    actual = effective(result, 2)["layer."].dense()
    singular = torch.linalg.svdvals(target)
    expected_error = float(singular[3:].square().sum())
    assert info["svd_tail_squared"] == pytest.approx(expected_error, rel=1e-8)
    assert float((actual - target).square().sum()) == pytest.approx(expected_error, rel=1e-6)


def test_svd_removes_cancelled_factor_directions():
    matrix = effective(state(1), 2)["layer."]
    cancelled = Matrix(torch.cat([matrix.u, -matrix.u], 1), torch.cat([matrix.v, matrix.v], 0))
    _, singular, _ = cancelled.svd()
    assert float(singular.max()) < 1e-12


def test_zero_weight_root_fallback_does_not_alias():
    base, local, root = state(0), state(1), state(2)
    result, info = aggregate(base, [local], [0], root, 2)
    for key in root:
        torch.testing.assert_close(result[key].double(), root[key], rtol=1e-6, atol=1e-6)
        assert result[key].data_ptr() != root[key].data_ptr()
    assert info["fallback"] == "root_only"
