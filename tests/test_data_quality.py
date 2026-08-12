"""
Unit + data tests for src/prepare.py (Week 6: unit testing for ML code and
data testing). Uses small synthetic DataFrames -- CI has no access to the
real DVC-tracked data/raw/data.csv (see docs/dataset.md and
docs/deployment.md for why no DVC remote is configured).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare  # noqa: E402


def test_clean_columns_fixes_known_quirks():
    df = pd.DataFrame(
        {
            "Daytime/evening attendance\t": [1],
            "Nacionality": [1],
            "Target": ["Graduate"],
        }
    )
    cleaned = prepare.clean_columns(df)
    assert "Daytime/evening attendance" in cleaned.columns
    assert "Nationality" in cleaned.columns
    assert "Nacionality" not in cleaned.columns


def test_handle_missing_imputes_numeric_with_median():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
    result = prepare.handle_missing(df.copy())
    assert result["a"].isna().sum() == 0
    assert result["a"].iloc[1] == 2.0  # median of [1, 3]
    assert result["b"].isna().sum() == 0


def test_handle_missing_is_noop_on_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = prepare.handle_missing(df.copy())
    pd.testing.assert_frame_equal(result, df)


def test_label_map_covers_all_three_classes():
    assert set(prepare.LABEL_MAP.keys()) == {"Dropout", "Enrolled", "Graduate"}
    assert set(prepare.LABEL_MAP.values()) == {0, 1, 2}


@pytest.mark.parametrize("label", ["Dropout", "Enrolled", "Graduate"])
def test_label_map_only_accepts_known_labels(label):
    df = pd.DataFrame({"Target": [label]})
    mapped = df["Target"].map(prepare.LABEL_MAP)
    assert mapped.isna().sum() == 0


def test_unknown_label_maps_to_nan_and_would_fail_the_assert():
    # Mirrors the assertion in prepare.main(): an unexpected label must be
    # caught, not silently coerced into a valid class index.
    df = pd.DataFrame({"Target": ["Unknown"]})
    mapped = df["Target"].map(prepare.LABEL_MAP)
    assert mapped.isna().sum() == 1
