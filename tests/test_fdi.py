"""Tests for the FDI tooth-number geometry estimator.

The synthetic arch below is laid out the way FDI numbers a panoramic image: the
patient's right side is on the left of the image, tooth 1 sits next to the
midline and tooth 8 sits at the back of the jaw.
"""

import json
from pathlib import Path

import pytest

import fdi

MIDLINE = 0.5
ARCH_LEVEL = 0.48
PITCH = 0.05
UPPER_V = 0.30
LOWER_V = 0.66


def synthetic_records():
    rows = []
    for quadrant, (v, side) in {1: (UPPER_V, -1), 2: (UPPER_V, +1), 3: (LOWER_V, +1), 4: (LOWER_V, -1)}.items():
        for tooth in range(1, 9):
            u = MIDLINE + side * (PITCH * (tooth - 1) + 0.02)
            jitter = ((tooth * 7 + quadrant * 3) % 5 - 2) * 0.002
            rows.append({"u": round(u + jitter, 6), "v": v, "quadrant": quadrant, "tooth": tooth})
    return rows * 12


@pytest.fixture(scope="module")
def table():
    return fdi.fit_table(synthetic_records(), v_weight=1.0, source={"fit_split": "train (synthetic)"})


def test_quadrant_of_follows_fdi_convention():
    assert fdi.quadrant_of(0.20, 0.30, MIDLINE, ARCH_LEVEL) == 1
    assert fdi.quadrant_of(0.80, 0.30, MIDLINE, ARCH_LEVEL) == 2
    assert fdi.quadrant_of(0.80, 0.70, MIDLINE, ARCH_LEVEL) == 3
    assert fdi.quadrant_of(0.20, 0.70, MIDLINE, ARCH_LEVEL) == 4


def test_fit_records_split_and_support(table):
    assert table["version"] == fdi.TABLE_VERSION
    assert table["basis"] == fdi.BASIS
    assert abs(table["quadrant_split"]["u_midline"] - MIDLINE) < 0.02
    assert abs(table["quadrant_split"]["v_arch"] - ARCH_LEVEL) < 0.02
    assert table["split_train_accuracy"] == {"side": 1.0, "level": 1.0}
    assert table["n_fit_boxes"] == len(synthetic_records())
    assert all(table["support"][f"{q}{t}"] == 12 for q in (1, 2, 3, 4) for t in range(1, 9))


def test_fit_recovers_every_tooth_digit(table):
    for row in synthetic_records():
        guess = fdi.assign_fdi(row["u"], row["v"], table)
        assert guess is not None
        assert guess["quadrant"] == row["quadrant"]
        assert guess["tooth"] == row["tooth"]
        assert guess["fdi"] == row["quadrant"] * 10 + row["tooth"]


def test_assignment_is_monotone_along_the_arch(table):
    upper_right = [fdi.assign_fdi(0.50 - 0.05 * k, UPPER_V, table)["fdi"] for k in range(8)]
    assert upper_right == [11, 12, 13, 14, 15, 16, 17, 18]
    lower_left = [fdi.assign_fdi(0.52 + 0.05 * k, LOWER_V, table)["fdi"] for k in range(8)]
    assert lower_left == [31, 32, 33, 34, 35, 36, 37, 38]


def test_margin_reports_a_decisive_assignment(table):
    centre = table["tooth_centroids"]["1"]["6"]
    decisive = fdi.assign_fdi(centre[0], centre[1], table)
    between = table["tooth_centroids"]["1"]["6"][0] * 0.5 + table["tooth_centroids"]["1"]["7"][0] * 0.5
    ambiguous = fdi.assign_fdi(between, centre[1], table)
    assert decisive["margin"] > ambiguous["margin"]
    assert 0.0 <= ambiguous["margin"] <= 1.0


def test_quadrant_without_support_returns_none():
    three_only = [row for row in synthetic_records() if row["quadrant"] != 4]
    partial = fdi.fit_table(three_only)
    assert fdi.assign_fdi(0.20, LOWER_V, partial) is None
    assert fdi.assign_fdi(0.20, UPPER_V, partial)["quadrant"] == 1


def test_estimate_fdi_uses_original_pixel_space(table):
    # Centre (0.28, 0.32) is the fifth tooth from the midline in the upper-right quadrant.
    box = [0.26 * 2000, UPPER_V * 1000, 0.30 * 2000, (UPPER_V + 0.04) * 1000]
    guess = fdi.estimate_fdi(box, 2000, 1000, table)
    assert guess["fdi"] == 15
    assert fdi.estimate_fdi(box, 2000, 1000, None) is None
    assert fdi.estimate_fdi(box, 0, 0, table) is None


def test_box_centre_clips_into_the_image():
    assert fdi.box_centre([-50, -50, 50, 50], 1000, 1000) == (0.0, 0.0)
    assert fdi.box_centre([900, 900, 1400, 1400], 1000, 1000) == (1.0, 1.0)


def test_load_table_rejects_missing_and_stale(tmp_path: Path):
    assert fdi.load_table(tmp_path / "absent.json") is None
    (tmp_path / "stale.json").write_text(json.dumps({"version": 99}), encoding="utf-8")
    assert fdi.load_table(tmp_path / "stale.json") is None
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert fdi.load_table(tmp_path / "broken.json") is None


def test_table_round_trips_through_disk(tmp_path: Path, table):
    path = tmp_path / "fdi_geometry.json"
    fdi.save_table(table, path)
    assert fdi.load_table(path) == table


def test_table_holds_only_fitted_summary_not_the_fitting_boxes(table):
    assert set(table) == {
        "version",
        "basis",
        "v_weight",
        "quadrant_split",
        "split_train_accuracy",
        "tooth_centroids",
        "support",
        "n_fit_boxes",
        "source",
    }
    assert table["n_fit_boxes"] == len(synthetic_records())
    assert sum(len(v) for v in table["tooth_centroids"].values()) <= 32


@pytest.mark.skipif(not fdi.TABLE_PATH.is_file(), reason="run scripts/dentex_fdi.py fit")
def test_shipped_table_is_fitted_on_train_and_measured_on_validation():
    shipped = fdi.load_table()
    assert shipped is not None
    assert "train" in shipped["source"]["fit_split"]
    assert "test" not in json.dumps(shipped["source"]).lower()
    assert shipped["tuning"]["selected_on"].startswith("a 1/5 image holdout")
    assert shipped["validation"]["n_boxes"] > 0
    assert 0.0 <= shipped["validation"]["fdi_accuracy"] <= 1.0
    guess = fdi.assign_fdi(0.25, 0.35, shipped)
    assert guess is None or 11 <= guess["fdi"] <= 48
