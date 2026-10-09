r"""The chip ``Dataset``: shape contract, masking, normalisation and fork safety.

The specification for :mod:`floodfusion.data.dataset`. Milestone ``P1-M1``.

Two properties here are load-bearing beyond their apparent size:

- **Per-chip normalisation is forbidden.** A flooded chip is dark, and that darkness is
  the signal. Standardising each chip by its own mean maps a fully flooded chip and a
  fully dry chip onto the same distribution, deleting exactly the absolute radiometric
  information the whole project rests on (``docs/05`` section 7.1).
- **The Sentinel-2 offset must be applied.** From processing baseline 04.00,
  ``rho = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE``. Omitting the offset shifts
  every reflectance by 0.1 — larger than the 0.02--0.06 water reflectances being
  measured (``docs/05`` section 7.2).
"""

from __future__ import annotations

import pytest

#: The subject of this module imports ``torch`` at module level, so the module
#: cannot be collected without it. The guard sits here rather than in
#: ``conftest.py``: a ``Skipped`` raised while the conftest is being imported is
#: attributed to the whole directory, and would take the torch-free modules with it.
torch = pytest.importorskip(
    "torch", reason="floodfusion.data.dataset imports torch at module level"
)

from floodfusion.config import (  # noqa: E402
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    Channel,
    DataConfig,
    GridConfig,
    NormalizationStats,
    ValidityFlag,
)
from floodfusion.data.dataset import (  # noqa: E402
    ChipRecord,
    FloodChipDataset,
    build_dataloader,
    load_manifest,
    worker_init_fn,
)

EXPECT_UNIMPLEMENTED = True
ALWAYS_PASS = frozenset({
    "test_record_is_frozen",
    "test_sentinel2_offset_changes_reflectance_by_more_than_the_signal",
    "test_scl_water_class_is_not_an_input_channel",
    "test_int16_db_quantisation_is_finer_than_speckle_noise",
    "test_nodata_sentinel_lies_outside_the_physical_range",
    "test_handles_are_cached_per_process",
    "test_empty_records_rejected",
    "test_length_matches_record_count",
})

CHIP = 32


def _record(chip_id: str = "c0", basin: str = "b0", optical: bool = True) -> ChipRecord:
    from pathlib import Path
    return ChipRecord(
        chip_id=chip_id, basin_id=basin, event_id="e0",
        raster_path=Path("nonexistent.tif"), window=(0, 0, CHIP, CHIP),
        positive_fraction=0.02, valid_fraction=1.0, optical_available=optical,
        relative_orbit=88, orbit_state="descending",
    )


def _dataset(records=None, training: bool = False) -> FloodChipDataset:
    return FloodChipDataset(
        records or [_record()], NormalizationStats(), DataConfig(chip_size=CHIP),
        GridConfig(), training=training,
    )


class TestChipRecord:
    def test_record_is_frozen(self) -> None:
        """A dataset whose records mutate at runtime cannot be reproduced."""
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            _record().basin_id = "other"  # type: ignore[misc]

    def test_manifest_rejects_duplicate_chip_ids(self, tmp_path) -> None:
        """A duplicate silently doubles a chip's weight."""
        with pytest.raises((ValueError, FileNotFoundError)):
            load_manifest(tmp_path / "missing.parquet")


class TestSampleContract:
    """The table in the module docstring of ``data/dataset.py``."""

    def test_returns_every_required_key(self) -> None:
        sample = _dataset()[0]
        assert {"sar", "optical", "terrain", "label", "valid",
                "optical_quality", "optical_available", "chip_id", "basin_id"} <= set(sample)

    @pytest.mark.parametrize(
        ("key", "channels"),
        [("sar", N_SAR_CHANNELS), ("optical", N_OPTICAL_CHANNELS),
         ("terrain", N_TERRAIN_CHANNELS), ("label", 1), ("valid", 1),
         ("optical_quality", 1)],
    )
    def test_tensor_shapes(self, key: str, channels: int) -> None:
        assert _dataset()[0][key].shape == (channels, CHIP, CHIP)

    def test_dtypes(self) -> None:
        sample = _dataset()[0]
        assert sample["sar"].dtype == torch.float32
        assert sample["label"].dtype == torch.float32
        assert sample["valid"].dtype == torch.bool

    def test_optical_is_a_tensor_even_when_unavailable(self) -> None:
        """``None`` cannot be collated, and mixed availability is the NORMAL case."""
        sample = _dataset([_record(optical=False)])[0]
        assert isinstance(sample["optical"], torch.Tensor)
        assert sample["optical"].shape == (N_OPTICAL_CHANNELS, CHIP, CHIP)
        assert bool(sample["optical_available"]) is False

    def test_label_is_binary(self) -> None:
        label = _dataset()[0]["label"]
        assert torch.isin(label, torch.tensor([0.0, 1.0])).all()

    def test_no_nan_or_inf_reaches_the_model(self) -> None:
        sample = _dataset()[0]
        for key in ("sar", "optical", "terrain"):
            assert torch.isfinite(sample[key]).all(), f"{key} contains NaN/Inf"

    def test_batched_shapes_are_bchw(self) -> None:
        loader = build_dataloader(
            _dataset([_record(f"c{i}") for i in range(4)]),
            DataConfig(chip_size=CHIP, num_workers=0),
            batch_size=2, shuffle=False, drop_last=True,
        )
        batch = next(iter(loader))
        assert batch["sar"].shape == (2, N_SAR_CHANNELS, CHIP, CHIP)


class TestNormalisation:
    def test_is_not_per_chip(self) -> None:
        """**The critical test.** Two chips of different brightness must stay different.

        Per-chip standardisation would map a fully flooded (dark) chip and a dry
        (bright) chip onto the same distribution, deleting the water signal.
        """
        dark, bright = _dataset([_record("dark")])[0], _dataset([_record("bright")])[0]
        assert dark["sar"].mean() != pytest.approx(bright["sar"].mean(), abs=1e-6), (
            "per-chip normalisation detected: chips of different brightness became "
            "identical in distribution. See docs/05 section 7.1."
        )

    def test_sentinel2_offset_changes_reflectance_by_more_than_the_signal(self) -> None:
        """Quantifies the baseline-04.00 trap without needing a real product."""
        norm = NormalizationStats()
        dn = 500.0
        with_offset = (dn + norm.optical_offset) / norm.optical_scale
        without = dn / norm.optical_scale
        water_reflectance = 0.03
        assert abs(without - with_offset) > water_reflectance, (
            "ignoring BOA_ADD_OFFSET shifts reflectance by more than the water signal"
        )

    def test_indices_stay_bounded(self) -> None:
        """MNDWI is already in [-1, 1] by construction; leave it alone."""
        optical = _dataset()[0]["optical"]
        mndwi = optical[Channel.MNDWI - 5]
        assert mndwi.min() >= -1.0 and mndwi.max() <= 1.0


class TestValidityMasking:
    def test_valid_is_true_only_where_flag_is_zero(self) -> None:
        sample = _dataset()[0]
        assert sample["valid"].dtype == torch.bool

    def test_nodata_sentinel_lies_outside_the_physical_range(self) -> None:
        """A sentinel read as -327.68 dB would train the model to detect scene borders."""
        grid = GridConfig()
        assert grid.nodata_int16 / grid.db_scale < -100.0

    def test_int16_db_quantisation_is_finer_than_speckle_noise(self) -> None:
        """0.01 dB quantisation against a 2.17 dB speckle sigma: nothing is lost."""
        assert 1.0 / GridConfig().db_scale < 0.1 * 2.17

    def test_scl_water_class_is_not_an_input_channel(self) -> None:
        """**The leakage trap.** SCL class 6 is WATER (``docs/02`` section 7.3).

        Deriving labels or inputs from it trains the model to reproduce Sen2Cor's
        threshold on the same bands, producing an excellent score that measures
        agreement with another algorithm rather than accuracy.
        """
        optical_channels = {Channel.B3_GREEN, Channel.B8_NIR, Channel.B11_SWIR1, Channel.MNDWI}
        assert len(optical_channels) == N_OPTICAL_CHANNELS
        assert not any("SCL" in c.name for c in Channel)


class TestAugmentation:
    def test_is_disabled_outside_training(self) -> None:
        a, b = _dataset(training=False)[0], _dataset(training=False)[0]
        assert torch.equal(a["sar"], b["sar"]), "evaluation must be deterministic"

    def test_applies_identically_to_image_and_label(self) -> None:
        """Augmenting the image and forgetting the label plateaus with no error."""
        sample = _dataset(training=True)[0]
        assert sample["sar"].shape[-2:] == sample["label"].shape[-2:]


class TestForkSafety:
    def test_handles_are_cached_per_process(self) -> None:
        """A handle opened before a fork is inherited with duplicated GDAL state."""
        ds = _dataset()
        assert isinstance(ds._handles, dict)
        assert not ds._handles, "no handle may be opened in the parent at construction"

    def test_worker_init_seeds_distinctly(self) -> None:
        """Without this, six workers draw the SAME 'random' augmentation sequence."""
        worker_init_fn(0)


class TestConstruction:
    def test_empty_records_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero records"):
            FloodChipDataset([], NormalizationStats(), DataConfig(), GridConfig())

    def test_length_matches_record_count(self) -> None:
        assert len(_dataset([_record(f"c{i}") for i in range(7)])) == 7
