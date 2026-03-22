"""Tests for Prometheus response formatting in tools/executor.py."""
from __future__ import annotations

import json

import pytest

from src.tools.executor import (
    format_prometheus_response,
    _format_metric_labels,
    _format_vector,
    _format_matrix,
    _PROM_MAX_RESULTS,
)


# ---------------------------------------------------------------------------
# _format_metric_labels
# ---------------------------------------------------------------------------

class TestFormatMetricLabels:
    def test_name_only(self):
        metric = {"__name__": "up"}
        assert _format_metric_labels(metric) == "up"

    def test_name_with_labels(self):
        metric = {"__name__": "up", "instance": "localhost:9090", "job": "prom"}
        result = _format_metric_labels(metric)
        assert result.startswith("up{")
        assert 'instance="localhost:9090"' in result
        assert 'job="prom"' in result

    def test_no_name(self):
        metric = {"instance": "localhost:9090"}
        result = _format_metric_labels(metric)
        assert result.startswith("{")
        assert 'instance="localhost:9090"' in result

    def test_empty_metric(self):
        assert _format_metric_labels({}) == ""

    def test_labels_sorted(self):
        metric = {"__name__": "m", "z_label": "1", "a_label": "2"}
        result = _format_metric_labels(metric)
        assert result.index("a_label") < result.index("z_label")

    def test_does_not_mutate_input(self):
        """_format_metric_labels must not mutate the input dict."""
        metric = {"__name__": "up", "job": "prom"}
        _format_metric_labels(metric)
        # __name__ should still be in the input dict after formatting
        assert "__name__" in metric
        assert metric["__name__"] == "up"
        assert metric["job"] == "prom"

    def test_name_only_no_extra_braces(self):
        """Metric with only __name__ and no labels should return just the name."""
        metric = {"__name__": "up"}
        result = _format_metric_labels(metric)
        assert result == "up"
        # Should not include __name__ as a label
        assert "__name__" not in result.replace("up", "")


# ---------------------------------------------------------------------------
# _format_vector
# ---------------------------------------------------------------------------

class TestFormatVector:
    def test_empty(self):
        assert _format_vector([]) == "No results."

    def test_single_result(self):
        results = [
            {"metric": {"__name__": "up", "job": "prom"}, "value": [1645000000, "1"]},
        ]
        text = _format_vector(results)
        assert "1 result:" in text
        assert "up{" in text
        assert ": 1" in text

    def test_multiple_results(self):
        results = [
            {"metric": {"__name__": "up", "instance": "a"}, "value": [0, "1"]},
            {"metric": {"__name__": "up", "instance": "b"}, "value": [0, "0"]},
        ]
        text = _format_vector(results)
        assert "2 results:" in text
        assert "a" in text
        assert "b" in text

    def test_truncation_at_max(self):
        results = [
            {"metric": {"__name__": f"m_{i}"}, "value": [0, str(i)]}
            for i in range(_PROM_MAX_RESULTS + 10)
        ]
        text = _format_vector(results)
        assert f"total: {_PROM_MAX_RESULTS + 10}" in text
        assert "... and 10 more" in text

    def test_missing_value_key(self):
        results = [{"metric": {"__name__": "m"}}]
        text = _format_vector(results)
        # Should not crash, shows fallback
        assert "m:" in text

    def test_value_format_fallback(self):
        """If value is not [ts, val], falls back to str()."""
        results = [{"metric": {"__name__": "m"}, "value": "unexpected"}]
        text = _format_vector(results)
        assert "unexpected" in text


# ---------------------------------------------------------------------------
# _format_matrix
# ---------------------------------------------------------------------------

class TestFormatMatrix:
    def test_empty(self):
        assert _format_matrix([]) == "No results."

    def test_single_series_multiple_points(self):
        results = [{
            "metric": {"__name__": "cpu", "mode": "idle"},
            "values": [[1000, "10"], [1300, "20"], [1600, "30"]],
        }]
        text = _format_matrix(results)
        assert "1 series:" in text
        assert "3 points" in text
        assert "10" in text  # first
        assert "30" in text  # last
        assert "\u2192" in text  # arrow

    def test_single_point(self):
        results = [{
            "metric": {"__name__": "m"},
            "values": [[1000, "42"]],
        }]
        text = _format_matrix(results)
        assert "1 point" in text
        assert "value=42" in text

    def test_zero_points(self):
        results = [{"metric": {"__name__": "m"}, "values": []}]
        text = _format_matrix(results)
        assert "0 points" in text

    def test_truncation_at_max(self):
        results = [
            {"metric": {"__name__": f"m_{i}"}, "values": [[0, "1"]]}
            for i in range(_PROM_MAX_RESULTS + 5)
        ]
        text = _format_matrix(results)
        assert f"total: {_PROM_MAX_RESULTS + 5}" in text
        assert "... and 5 more" in text


# ---------------------------------------------------------------------------
# format_prometheus_response — full integration
# ---------------------------------------------------------------------------

class TestFormatPrometheusResponse:
    def test_valid_vector_json(self):
        raw = json.dumps({
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "up", "job": "prom"}, "value": [1645000000, "1"]},
                    {"metric": {"__name__": "up", "job": "node"}, "value": [1645000000, "1"]},
                ],
            },
        })
        text = format_prometheus_response(raw)
        assert "2 results:" in text
        assert "up{" in text
        assert '"status"' not in text  # no raw JSON

    def test_valid_scalar_json(self):
        raw = json.dumps({
            "status": "success",
            "data": {"resultType": "scalar", "result": [1645000000, "42.5"]},
        })
        text = format_prometheus_response(raw)
        assert text == "Result: 42.5"

    def test_valid_string_json(self):
        raw = json.dumps({
            "status": "success",
            "data": {"resultType": "string", "result": [1645000000, "hello"]},
        })
        text = format_prometheus_response(raw)
        assert text == "Result: hello"

    def test_valid_matrix_json(self):
        raw = json.dumps({
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"__name__": "cpu_total"},
                    "values": [[1000, "100"], [2000, "200"]],
                }],
            },
        })
        text = format_prometheus_response(raw)
        assert "1 series:" in text
        assert "2 points" in text
        assert "100" in text
        assert "200" in text

    def test_error_status(self):
        raw = json.dumps({"status": "error", "error": "bad query syntax"})
        text = format_prometheus_response(raw)
        assert "Prometheus error" in text
        assert "bad query syntax" in text

    def test_error_without_message(self):
        raw = json.dumps({"status": "error"})
        text = format_prometheus_response(raw)
        assert "Prometheus error" in text

    def test_invalid_json_returns_raw(self):
        raw = "this is not json"
        assert format_prometheus_response(raw) == raw

    def test_empty_string_returns_raw(self):
        assert format_prometheus_response("") == ""

    def test_partial_json_returns_raw(self):
        raw = '{"status": "suc'
        assert format_prometheus_response(raw) == raw

    def test_unknown_result_type_returns_raw(self):
        raw = json.dumps({
            "status": "success",
            "data": {"resultType": "unknown_type", "result": []},
        })
        assert format_prometheus_response(raw) == raw

    def test_empty_vector_result(self):
        raw = json.dumps({
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        })
        text = format_prometheus_response(raw)
        assert text == "No results."

    def test_empty_matrix_result(self):
        raw = json.dumps({
            "status": "success",
            "data": {"resultType": "matrix", "result": []},
        })
        text = format_prometheus_response(raw)
        assert text == "No results."

    def test_none_input_returns_raw(self):
        """None input should not crash."""
        result = format_prometheus_response(None)
        assert result is None

    def test_large_vector_is_concise(self):
        """A 100-result vector should be significantly shorter than raw JSON."""
        results = [
            {
                "metric": {"__name__": f"metric_{i}", "instance": f"host{i}:9100"},
                "value": [1645000000, str(i * 1.5)],
            }
            for i in range(100)
        ]
        raw = json.dumps({
            "status": "success",
            "data": {"resultType": "vector", "result": results},
        })
        formatted = format_prometheus_response(raw)
        # Formatted should be much shorter than raw JSON
        assert len(formatted) < len(raw) * 0.6
        # Should show truncation notice
        assert "... and 50 more" in formatted

    def test_large_matrix_is_concise(self):
        """A range query with many data points should be summarised."""
        values = [[1645000000 + i * 300, str(i)] for i in range(200)]
        raw = json.dumps({
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"__name__": "cpu_total", "cpu": "0"},
                    "values": values,
                }],
            },
        })
        formatted = format_prometheus_response(raw)
        # Should be dramatically shorter than 200 data points
        assert len(formatted) < len(raw) * 0.15
        assert "200 points" in formatted

    def test_preserves_metric_values_accurately(self):
        """Spot-check that specific metric values appear in formatted output."""
        raw = json.dumps({
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "node_load1", "instance": "server:9100"}, "value": [0, "2.34"]},
                ],
            },
        })
        text = format_prometheus_response(raw)
        assert "node_load1" in text
        assert "server:9100" in text
        assert "2.34" in text
