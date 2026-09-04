from __future__ import annotations

import math

import pytest

from ttanco.dataset import generate_instance
from ttanco.domain import (
    TSPInstance,
    audit_tour,
    canonicalize_tour,
    nearest_neighbor_tour,
    solve_brute_force,
    solve_held_karp,
    solution_from_tour,
    two_opt,
)


def square() -> TSPInstance:
    return TSPInstance(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))


def test_canonical_tour_and_square_oracles_agree() -> None:
    instance = square()
    assert canonicalize_tour((2, 3, 0, 1)) == (0, 1, 2, 3)
    assert canonicalize_tour((0, 3, 2, 1)) == (0, 1, 2, 3)
    dynamic = solve_held_karp(instance)
    brute = solve_brute_force(instance)
    assert dynamic.tour == (0, 1, 2, 3)
    assert dynamic.length == pytest.approx(4.0)
    assert brute.length == pytest.approx(dynamic.length)


def test_random_held_karp_matches_brute_force() -> None:
    for seed in range(4):
        instance = generate_instance(7, regime="uniform", seed=seed)
        dynamic = solve_held_karp(instance)
        brute = solve_brute_force(instance)
        assert dynamic.length == pytest.approx(brute.length, rel=1e-10, abs=1e-10)
        audit = audit_tour(instance, dynamic.tour, reported_length=dynamic.length, optimum=brute)
        assert audit.permutation_valid
        assert audit.optimal is True


def test_audit_rejects_malformed_tour_and_inconsistent_length() -> None:
    instance = square()
    invalid = audit_tour(instance, (0, 1, 1, 3))
    assert not invalid.permutation_valid
    solution = solution_from_tour(instance, (0, 1, 2, 3))
    inconsistent = audit_tour(instance, solution.tour, reported_length=99.0)
    assert inconsistent.permutation_valid
    assert not inconsistent.reported_length_consistent


def test_two_opt_improves_crossed_tour_and_never_breaks_feasibility() -> None:
    instance = square()
    crossed = solution_from_tour(instance, (0, 2, 1, 3))
    improved = two_opt(instance, crossed.tour)
    assert improved.solution.length < crossed.length
    assert improved.solution.length == pytest.approx(4.0)
    assert improved.move_evaluations > 0
    assert nearest_neighbor_tour(instance).length == pytest.approx(4.0)


def test_domain_validation_and_exact_limits() -> None:
    with pytest.raises(ValueError):
        TSPInstance(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError):
        TSPInstance(((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    instance = generate_instance(6, seed=9)
    with pytest.raises(ValueError):
        solve_held_karp(instance, maximum_nodes=5)
    with pytest.raises(ValueError):
        solve_brute_force(instance, maximum_nodes=5)
    with pytest.raises(ValueError):
        canonicalize_tour((0, 1, 2), 4)


def test_positive_coordinate_scaling_preserves_tour_order_and_scales_length() -> None:
    base = generate_instance(6, seed=17)
    scaled = TSPInstance(
        tuple((10.0 * x + 5.0, 10.0 * y - 7.0) for x, y in base.coordinates)
    )
    base_optimum = solve_held_karp(base)
    scaled_optimum = solve_held_karp(scaled)
    assert scaled_optimum.tour == base_optimum.tour
    assert scaled_optimum.length == pytest.approx(10.0 * base_optimum.length)
    assert math.isfinite(scaled_optimum.length)
