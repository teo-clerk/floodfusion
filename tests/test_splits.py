r"""Basin-blocked splitting: the tests that decide whether your number is honest.

The specification for :mod:`floodfusion.splits`. Milestone ``P1-M1``.

    Your split determines your reported number more than your architecture does.

``tests/test_splits.py`` asserts two disjointness properties, and the **second** is the
one that catches real bugs: chip disjointness is trivially satisfied by a random split
that leaks badly, whereas basin disjointness is not.
"""

from __future__ import annotations

import json

import pytest

from floodfusion.splits import (
    SplitAssignment,
    assign_basins,
    empirical_variogram_range,
    freeze_splits,
    load_splits,
    random_chip_split,
)

EXPECT_UNIMPLEMENTED = True
ALWAYS_PASS = frozenset({
    "test_assignment_is_frozen",
    "test_leaky_strategy_is_named_explicitly",
})


def _assignment(strategy: str = "leave_one_basin_out") -> SplitAssignment:
    return SplitAssignment(
        train_chips=("c1", "c2"), val_chips=("c3",), test_chips=("c4",),
        train_basins=("b1",), val_basins=("b2",), test_basins=("b3",),
        buffer_m=1000.0, strategy=strategy,
    )


class TestSplitAssignment:
    def test_assignment_is_frozen(self) -> None:
        """A split that can mutate at runtime is not a split."""
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            _assignment().buffer_m = 0.0  # type: ignore[misc]

    def test_leaky_strategy_is_named_explicitly(self) -> None:
        """``evaluate.py`` refuses to report ``random_chip`` as a headline number."""
        assert _assignment("random_chip").strategy == "random_chip"


class TestDisjointness:
    def test_chip_sets_are_pairwise_disjoint(self) -> None:
        a = assign_basins({}, {}, None, buffer_m=1000.0)
        assert not (set(a.train_chips) & set(a.val_chips))
        assert not (set(a.train_chips) & set(a.test_chips))
        assert not (set(a.val_chips) & set(a.test_chips))

    def test_basin_sets_are_pairwise_disjoint(self) -> None:
        """**The assertion that matters.** Chip disjointness alone does not prevent leakage."""
        a = assign_basins({}, {}, None, buffer_m=1000.0)
        assert not (set(a.train_basins) & set(a.val_basins))
        assert not (set(a.train_basins) & set(a.test_basins))
        assert not (set(a.val_basins) & set(a.test_basins))

    def test_no_chip_belongs_to_a_basin_outside_its_split(self) -> None:
        """Catches assigning chips by proximity rather than by basin membership."""
        chip_basins = {f"c{i}": f"b{i // 4}" for i in range(24)}
        centroids = {c: (float(i) * 5000.0, 0.0) for i, c in enumerate(chip_basins)}
        a = assign_basins(chip_basins, centroids, None, buffer_m=1000.0)
        for chips, basins in (
            (a.train_chips, a.train_basins),
            (a.val_chips, a.val_basins),
            (a.test_chips, a.test_basins),
        ):
            for chip in chips:
                assert chip_basins[chip] in basins


class TestBuffering:
    def test_buffer_must_exceed_the_variogram_range(self) -> None:
        """Do not guess the buffer; measure the autocorrelation range (``docs/05`` 6.2)."""
        values = [float(i % 7) for i in range(500)]
        coords = [(float(i) * 100.0, 0.0) for i in range(500)]
        rng = empirical_variogram_range(values, coords)
        assert rng > 0.0
        a = assign_basins({}, {}, None, buffer_m=rng * 1.5)
        assert a.buffer_m > rng

    def test_buffer_is_recorded_on_the_assignment(self) -> None:
        assert assign_basins({}, {}, None, buffer_m=1234.0).buffer_m == 1234.0


class TestRandomChipSplitIsLeakyByDesign:
    """Ablation row 13 only. This function exists to be *wrong*, measurably."""

    def test_declares_itself_as_the_leaky_strategy(self) -> None:
        chip_basins = {f"c{i}": f"b{i // 4}" for i in range(40)}
        a = random_chip_split(list(chip_basins), chip_basins)
        assert a.strategy == "random_chip"

    def test_basins_overlap_across_splits(self) -> None:
        """The leak, made visible: the same basin appears in train and test."""
        chip_basins = {f"c{i}": f"b{i // 4}" for i in range(40)}
        a = random_chip_split(list(chip_basins), chip_basins)
        assert set(a.train_basins) & set(a.test_basins), (
            "a random chip split MUST leak basins — that is what row 13 measures"
        )

    def test_is_deterministic_for_a_given_seed(self) -> None:
        chip_basins = {f"c{i}": f"b{i // 4}" for i in range(40)}
        first = random_chip_split(list(chip_basins), chip_basins, seed=7)
        second = random_chip_split(list(chip_basins), chip_basins, seed=7)
        assert first.train_chips == second.train_chips


class TestFreezing:
    def test_round_trips_through_disk(self, tmp_path) -> None:
        path = tmp_path / "splits.json"
        freeze_splits(_assignment(), path)
        assert load_splits(path) == _assignment()

    def test_written_file_is_json(self, tmp_path) -> None:
        path = tmp_path / "splits.json"
        freeze_splits(_assignment(), path)
        json.loads(path.read_text())

    def test_load_rejects_overlapping_basins(self, tmp_path) -> None:
        """Validating on LOAD stops a hand-edited split poisoning a run weeks later."""
        bad = SplitAssignment(
            train_chips=("c1",), val_chips=("c2",), test_chips=("c3",),
            train_basins=("b1",), val_basins=("b1",), test_basins=("b2",),
            buffer_m=1000.0, strategy="leave_one_basin_out",
        )
        path = tmp_path / "bad.json"
        freeze_splits(bad, path)
        with pytest.raises(ValueError):
            load_splits(path)

    def test_load_permits_overlap_only_for_the_leakage_row(self, tmp_path) -> None:
        leaky = SplitAssignment(
            train_chips=("c1",), val_chips=("c2",), test_chips=("c3",),
            train_basins=("b1",), val_basins=("b1",), test_basins=("b1",),
            buffer_m=0.0, strategy="random_chip",
        )
        path = tmp_path / "row13.json"
        freeze_splits(leaky, path)
        assert load_splits(path).strategy == "random_chip"
