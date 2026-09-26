"""The app runs from the committed curves.json alone and shows numbers consistent with results.json."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    return at


def test_runs_without_exceptions(app):
    assert not app.exception


def test_banner_present(app):
    assert any("Illustrative cost/value assumptions" in w.value for w in app.warning)


def test_default_optimum_matches_results(app):
    res = json.loads((ROOT / "results" / "results.json").read_text())["policy"]["default_scenario"]
    shown = {m.label: m.value for m in app.metric}
    assert shown["Profit-maximizing k"] == f"{100 * res['optimal_k']:.1f}%"


def test_slider_changes_policy(app):
    app.sidebar.slider[1].set_value(0.05).run()  # cost $0.05 at v=$1 -> c/v 0.05
    shown = {m.label: m.value for m in app.metric}
    assert shown["Profit-maximizing k"] == "3.5%"
    assert not app.exception


def test_app_never_reads_raw_data():
    src = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "curves.json" in src
    for forbidden in ("parquet", "data/raw", "results.json", "read_csv"):
        assert forbidden not in src
