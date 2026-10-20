import copy
import csv
import json

import pytest

from tricg import build_matrix, build_profile
from tricg.cli import main
from tricg.importers import stage1_activations


@pytest.fixture
def instrument():
    return {
        "model": "test-model@revision",
        "layer": 0,
        "sae": "test-dictionary@revision",
        "token_policy": "all-input-tokens",
    }


def test_profile_preserves_silent_features_and_relative_scale(instrument):
    profile = build_profile("a", [{0: 2, 2: 1}, {0: 1}], 4, instrument)
    assert profile["coverage"] == pytest.approx([3, 0, 1, 0])
    assert profile["item_count"] == 2
    assert profile["feature_count"] == 4
    scaled = build_profile("a", [{0: 20, 2: 10}, {0: 10}], 4, instrument)
    assert scaled["coverage"] == pytest.approx(profile["coverage"])


@pytest.mark.parametrize(
    "activations",
    [
        [],
        [{}],
        [{0: -1}],
        [{0: float("nan")}],
        [{0: float("inf")}],
        [{4: 1}],
        [{-1: 1}],
        [{"1.5": 1}],
    ],
)
def test_profile_rejects_invalid_activations(instrument, activations):
    with pytest.raises(ValueError):
        build_profile("a", activations, 4, instrument)


@pytest.mark.parametrize("width", [0, -1, 4.5, True])
def test_profile_requires_full_integer_dictionary_width(instrument, width):
    with pytest.raises(ValueError):
        build_profile("a", [{0: 1}], width, instrument)


def test_matrix_preserves_benchmark_and_concept_order(instrument):
    a = build_profile("a", [{0: 2, 2: 1}, {0: 1}], 4, instrument)
    b = build_profile("b", [{1: 1, 3: 1}], 4, instrument)
    matrix = build_matrix([b, a])
    assert matrix["benchmarks"] == ["b", "a"]
    assert matrix["concept_ids"] == [0, 1, 2, 3]
    assert matrix["coverage"] == [[0, 2, 0, 2], [3, 0, 1, 0]]


@pytest.mark.parametrize("change", ["instrument", "feature_count", "coverage", "benchmark"])
def test_matrix_rejects_incompatible_or_invalid_profiles(instrument, change):
    a = build_profile("a", [{0: 1}], 4, instrument)
    b = copy.deepcopy(a)
    b["benchmark"] = "b"
    if change == "instrument":
        b["instrument"]["sae"] = "other-dictionary"
    elif change == "feature_count":
        b["feature_count"] = 5
    elif change == "coverage":
        b["coverage"] = [1, 0, 0, 0]
    else:
        b["benchmark"] = "a"
    with pytest.raises(ValueError):
        build_matrix([a, b])


def test_csv_normalizes_each_item_before_aggregation(tmp_path, instrument):
    path = tmp_path / "stage1.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["concept_activations", "num_tokens"])
        writer.writeheader()
        writer.writerow({"concept_activations": '{"0": 20, "2": 10}', "num_tokens": 10})
        writer.writerow({"concept_activations": '{"0": 2}', "num_tokens": 2})
    profile = build_profile("a", stage1_activations(path), 4, instrument)
    assert profile["coverage"] == pytest.approx([3, 0, 1, 0])
    assert profile["item_count"] == 2


def test_csv_rejects_failed_extraction(tmp_path):
    path = tmp_path / "stage1.csv"
    path.write_text("concept_activations,num_tokens\n{},0\n")
    with pytest.raises(ValueError, match="no valid tokens"):
        list(stage1_activations(path))


def test_cli_profile_to_matrix(tmp_path, monkeypatch, capsys, instrument):
    source = tmp_path / "input.json"
    profile = tmp_path / "profile.json"
    source.write_text(
        json.dumps(
            {
                "benchmark": "a",
                "instrument": instrument,
                "feature_count": 4,
                "activations": [{"0": 2, "2": 1}, {"0": 1}],
                "source": {"synthetic": True},
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["tricg", "profile", str(source), "-o", str(profile)])
    main()
    assert json.loads(profile.read_text())["source"] == {"synthetic": True}
    monkeypatch.setattr("sys.argv", ["tricg", "matrix", str(profile)])
    main()
    assert json.loads(capsys.readouterr().out)["coverage"] == [[3, 0, 1, 0]]


def test_cli_reports_invalid_input_without_traceback(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.json"
    source.write_text("{}")
    monkeypatch.setattr("sys.argv", ["tricg", "profile", str(source)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "tricg error" in capsys.readouterr().err
