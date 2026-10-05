"""Tests for the SLO threshold evaluation module."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from httptap.models import ResponseInfo, StepMetrics, TimingMetrics
from httptap.slo import (
    SLO_KEYS,
    SLOBaselineError,
    SLOBudget,
    SLOResult,
    SLOScope,
    SLOSpecError,
    SLOTarget,
    SLOViolation,
    ensure_baseline_matches,
    evaluate_slo,
    evaluate_target,
    load_baseline,
    normalize_url,
    parse_slo_spec,
    parse_slo_target,
    select_step_for_evaluation,
)


def _step(
    *,
    step_number: int = 1,
    error: str | None = None,
    **timing_fields: float,
) -> StepMetrics:
    """Build a minimal ``StepMetrics`` for SLO tests.

    Any ``<key>_ms`` keyword argument is forwarded to
    :class:`TimingMetrics`; unknown keys raise ``TypeError`` from the
    dataclass constructor.

    """
    return StepMetrics(
        url="https://example.com",
        step_number=step_number,
        timing=TimingMetrics(**timing_fields),  # type: ignore[arg-type]
        error=error,
    )


class TestSLOKeys:
    """SLO_KEYS should cover every duration-valued TimingMetrics field."""

    def test_expected_keys_are_present(self) -> None:
        assert frozenset({"dns", "connect", "tls", "ttfb", "wait", "xfer", "total"}) == SLO_KEYS

    def test_does_not_contain_is_estimated(self) -> None:
        assert "is_estimated" not in SLO_KEYS

    def test_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            SLO_KEYS.add("new_key")  # type: ignore[attr-defined]


class TestParseSLOSpec:
    """Parsing accepts well-formed specs and rejects anything else."""

    def test_single_key(self) -> None:
        assert parse_slo_spec("total=500") == {"total": 500.0}

    def test_multiple_keys(self) -> None:
        result = parse_slo_spec("total=500,ttfb=200,connect=100")
        assert result == {"total": 500.0, "ttfb": 200.0, "connect": 100.0}

    def test_fractional_values(self) -> None:
        assert parse_slo_spec("ttfb=123.45") == {"ttfb": 123.45}

    def test_whitespace_tolerance(self) -> None:
        assert parse_slo_spec(" total = 500 , ttfb = 200 ") == {
            "total": 500.0,
            "ttfb": 200.0,
        }

    def test_case_insensitive_keys(self) -> None:
        assert parse_slo_spec("TOTAL=500,Ttfb=200") == {"total": 500.0, "ttfb": 200.0}

    @pytest.mark.parametrize("key", sorted(SLO_KEYS))
    def test_every_supported_key(self, key: str) -> None:
        assert parse_slo_spec(f"{key}=100") == {key: 100.0}

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_empty_spec_rejected(self, raw: str) -> None:
        with pytest.raises(SLOSpecError, match="empty"):
            parse_slo_spec(raw)

    def test_empty_item_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="Empty item"):
            parse_slo_spec("total=500,,ttfb=200")

    @pytest.mark.parametrize("raw", ["total", "total=500=1000", "=500"])
    def test_malformed_item_rejected(self, raw: str) -> None:
        with pytest.raises(SLOSpecError):
            parse_slo_spec(raw)

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="Unknown SLO key 'foo'"):
            parse_slo_spec("foo=500")

    def test_duplicate_key_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="Duplicate SLO key 'total'"):
            parse_slo_spec("total=500,total=600")

    def test_non_numeric_value_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="is not a number"):
            parse_slo_spec("total=fast")

    @pytest.mark.parametrize("value", ["0", "-1", "-0.001", "inf", "nan"])
    def test_non_positive_or_non_finite_value_rejected(self, value: str) -> None:
        with pytest.raises(SLOSpecError, match="positive finite number"):
            parse_slo_spec(f"total={value}")

    def test_returns_new_dict_each_call(self) -> None:
        first = parse_slo_spec("total=500")
        second = parse_slo_spec("total=500")
        assert first == second
        assert first is not second


class TestEvaluateSLO:
    """Evaluation reports violations only when timings exceed budgets."""

    def test_pass_when_every_metric_under_budget(self) -> None:
        step = _step(total_ms=100.0, ttfb_ms=50.0)
        result = evaluate_slo(step, {"total": 500.0, "ttfb": 200.0})
        assert result.passed is True
        assert result.violations == ()
        assert dict(result.thresholds_ms) == {"total": 500.0, "ttfb": 200.0}

    def test_equal_to_threshold_is_pass(self) -> None:
        step = _step(total_ms=500.0)
        result = evaluate_slo(step, {"total": 500.0})
        assert result.passed is True

    def test_fail_when_single_metric_exceeds(self) -> None:
        step = _step(total_ms=600.0)
        result = evaluate_slo(step, {"total": 500.0})
        assert result.passed is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.key == "total"
        assert violation.threshold_ms == 500.0
        assert violation.actual_ms == 600.0
        assert violation.delta_ms == pytest.approx(100.0)

    def test_fail_reports_all_violations_in_sorted_order(self) -> None:
        step = _step(total_ms=600.0, ttfb_ms=300.0, connect_ms=200.0)
        result = evaluate_slo(
            step,
            {"ttfb": 200.0, "total": 500.0, "connect": 100.0},
        )
        assert result.passed is False
        # Sorted alphabetically for deterministic output.
        assert [v.key for v in result.violations] == ["connect", "total", "ttfb"]

    def test_empty_thresholds_always_pass(self) -> None:
        step = _step(total_ms=10000.0)
        result = evaluate_slo(step, {})
        assert result.passed is True
        assert dict(result.thresholds_ms) == {}

    @pytest.mark.parametrize("threshold_key", sorted(SLO_KEYS))
    def test_every_slo_key_maps_to_timing_field(self, threshold_key: str) -> None:
        """Every key in SLO_KEYS must resolve to a TimingMetrics attribute.

        Parametrised directly over :data:`SLO_KEYS` so that adding a
        new entry there breaks this test until :func:`evaluate_slo`
        grows the corresponding mapping.
        """
        actual_ms = 200.0
        timing = TimingMetrics(**{f"{threshold_key}_ms": actual_ms})  # type: ignore[arg-type]
        step = StepMetrics(url="https://example.com", timing=timing)

        # Threshold is half the measured value → violation on the one key.
        result = evaluate_slo(step, {threshold_key: actual_ms / 2})

        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].key == threshold_key
        assert result.violations[0].actual_ms == actual_ms

    def test_unknown_key_raises_spec_error(self) -> None:
        step = _step(total_ms=100.0)
        with pytest.raises(SLOSpecError, match="Unknown SLO key"):
            evaluate_slo(step, {"bogus": 500.0})

    def test_mixed_known_and_unknown_keys_rejected(self) -> None:
        step = _step(total_ms=100.0)
        with pytest.raises(SLOSpecError, match="bogus"):
            evaluate_slo(step, {"total": 500.0, "bogus": 500.0})

    def test_does_not_mutate_input_thresholds(self) -> None:
        step = _step(total_ms=600.0)
        thresholds = {"total": 500.0, "ttfb": 100.0}
        snapshot = dict(thresholds)

        evaluate_slo(step, thresholds)

        assert thresholds == snapshot  # Caller's mapping preserved.

    def test_zero_phase_against_zero_threshold_is_pass(self) -> None:
        """TLS=0 on plain HTTP should not violate a tight TLS budget."""
        step = _step(tls_ms=0.0)
        result = evaluate_slo(step, {"tls": 1.0})
        assert result.passed is True


class TestSLOResult:
    """``SLOResult.to_dict`` produces stable JSON-ready data."""

    def test_pass_dict_shape(self) -> None:
        result = SLOResult(thresholds_ms={"total": 500.0}, violations=())
        assert result.to_dict() == {
            "pass": True,
            "thresholds_ms": {"total": 500.0},
            "violations": [],
        }

    def test_fail_dict_shape(self) -> None:
        violation = SLOViolation(key="total", threshold_ms=500.0, actual_ms=700.0)
        result = SLOResult(thresholds_ms={"total": 500.0}, violations=(violation,))
        payload = result.to_dict()
        assert payload == {
            "pass": False,
            "thresholds_ms": {"total": 500.0},
            "violations": [
                {
                    "key": "total",
                    "threshold_ms": 500.0,
                    "actual_ms": 700.0,
                    "delta_ms": 200.0,
                }
            ],
        }

    def test_evaluate_copies_user_thresholds(self) -> None:
        """evaluate_slo snapshots the caller's mapping into the result."""
        thresholds = {"total": 500.0}
        step = _step(total_ms=100.0)

        result = evaluate_slo(step, thresholds)

        thresholds["total"] = 999.0
        assert result.thresholds_ms == {"total": 500.0}

    def test_frozen_attribute_reassignment_is_blocked(self) -> None:
        result = SLOResult(thresholds_ms={"total": 500.0}, violations=())
        with pytest.raises(AttributeError):
            result.violations = ()  # type: ignore[misc]

    def test_to_dict_thresholds_sorted_for_stable_json(self) -> None:
        """to_dict emits thresholds in alphabetical key order."""
        result = SLOResult(
            thresholds_ms={"total": 500.0, "connect": 100.0, "ttfb": 200.0},
            violations=(),
        )
        payload_keys = list(result.to_dict()["thresholds_ms"].keys())
        assert payload_keys == ["connect", "total", "ttfb"]

    def test_equality_compares_by_value(self) -> None:
        violation = SLOViolation(key="total", threshold_ms=500.0, actual_ms=700.0)
        first = SLOResult(thresholds_ms={"total": 500.0}, violations=(violation,))
        second = SLOResult(thresholds_ms={"total": 500.0}, violations=(violation,))
        assert first == second

    def test_is_not_hashable(self) -> None:
        result = SLOResult(thresholds_ms={"total": 500.0}, violations=())
        with pytest.raises(TypeError):
            hash(result)


class TestSLOViolation:
    """``SLOViolation.delta_ms`` captures the overrun."""

    def test_delta_is_positive_on_overrun(self) -> None:
        v = SLOViolation(key="total", threshold_ms=500.0, actual_ms=750.0)
        assert v.delta_ms == pytest.approx(250.0)

    def test_delta_is_zero_on_boundary(self) -> None:
        v = SLOViolation(key="total", threshold_ms=500.0, actual_ms=500.0)
        assert math.isclose(v.delta_ms, 0.0)

    def test_is_immutable(self) -> None:
        v = SLOViolation(key="total", threshold_ms=500.0, actual_ms=750.0)
        with pytest.raises(AttributeError):
            v.threshold_ms = 0.0  # type: ignore[misc]


class TestSelectStepForEvaluation:
    """SLO targets the final successful step of the chain."""

    def test_single_successful_step(self) -> None:
        step = _step(total_ms=100.0)
        assert select_step_for_evaluation([step]) is step

    def test_returns_last_step_when_all_succeed(self) -> None:
        first = _step(total_ms=50.0, step_number=1)
        second = _step(total_ms=100.0, step_number=2)
        third = _step(total_ms=150.0, step_number=3)
        assert select_step_for_evaluation([first, second, third]) is third

    def test_skips_trailing_errors(self) -> None:
        ok = _step(total_ms=100.0, step_number=1)
        failed = _step(step_number=2, error="connection refused")
        assert select_step_for_evaluation([ok, failed]) is ok

    def test_returns_none_when_all_failed(self) -> None:
        first = _step(step_number=1, error="DNS error")
        second = _step(step_number=2, error="TCP error")
        assert select_step_for_evaluation([first, second]) is None

    def test_empty_list_returns_none(self) -> None:
        assert select_step_for_evaluation([]) is None


# ---------------------------------------------------------------------------
# Helpers for the explicit target / baseline tests
# ---------------------------------------------------------------------------


def _chain_step(
    *,
    url: str,
    step_number: int,
    status: int,
    total_ms: float,
    method: str = "GET",
    **timing_fields: float,
) -> StepMetrics:
    """Build a fully-shaped chain step."""
    timing_fields.setdefault("total_ms", total_ms)
    return StepMetrics(
        url=url,
        step_number=step_number,
        timing=TimingMetrics(**timing_fields),  # type: ignore[arg-type]
        response=ResponseInfo(status=status),
        request_method=method,
    )


def _baseline_payload(hops: list[tuple[str, int, str, float]]) -> dict:
    """Build an httptap JSON report payload from (url, status, method, total)."""
    return {
        "initial_url": hops[0][0],
        "total_steps": len(hops),
        "steps": [
            {
                "url": url,
                "step_number": index + 1,
                "request": {"method": method, "headers": {}, "body_bytes": 0},
                "timing": {
                    "dns_ms": 0.0,
                    "connect_ms": 0.0,
                    "tls_ms": 0.0,
                    "ttfb_ms": 0.0,
                    "total_ms": total,
                    "wait_ms": 0.0,
                    "xfer_ms": 0.0,
                    "is_estimated": False,
                },
                "response": {"status": status},
                "error": None,
            }
            for index, (url, status, method, total) in enumerate(hops)
        ],
        "summary": {},
    }


def _write_report(path: Path, hops: list[tuple[str, int, str, float]]) -> Path:
    path.write_text(json.dumps(_baseline_payload(hops)), encoding="utf-8")
    return path


# The canonical redirect chain used by the acceptance scenarios.
CHAIN_HOPS = [
    ("https://hop.test/start", 301, "GET", 200.0),
    ("https://hop.test/final", 200, "GET", 250.0),
]


class TestParseSLOTarget:
    """The explicit parser handles scopes, budgets, and rejects mixing."""

    def test_bare_spec_is_final_absolute(self) -> None:
        target = parse_slo_target("total=500")
        assert target.scope is SLOScope.FINAL
        assert target.keys == ("total",)
        assert target.budgets[0].absolute_ms == 500.0
        assert target.budgets[0].is_relative is False

    def test_explicit_final_prefix(self) -> None:
        target = parse_slo_target("final.total=100")
        assert target.scope is SLOScope.FINAL

    def test_chain_prefix(self) -> None:
        target = parse_slo_target("chain.total=300")
        assert target.scope is SLOScope.CHAIN
        assert target.budgets[0].absolute_ms == 300.0

    def test_each_prefix(self) -> None:
        target = parse_slo_target("each.ttfb=100")
        assert target.scope is SLOScope.EACH

    def test_multiple_items_share_scope(self) -> None:
        target = parse_slo_target("chain.total=300,chain.ttfb=100")
        assert target.scope is SLOScope.CHAIN
        assert target.keys == ("total", "ttfb")

    def test_mixed_scopes_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="Mixed SLO scopes"):
            parse_slo_target("final.total=100,chain.total=300")

    def test_scoped_and_bare_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="Mix of scoped"):
            parse_slo_target("chain.total=300,ttfb=100")

    def test_unknown_scope_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="Unknown SLO scope 'whole'"):
            parse_slo_target("whole.total=300")

    def test_relative_without_baseline_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="requires --slo-baseline"):
            parse_slo_target("chain.total=+20%")

    @pytest.mark.parametrize("raw", ["+20%", "20%", "-5%"])
    def test_relative_values_with_baseline(self, raw: str) -> None:
        target = parse_slo_target(f"chain.total={raw}", baseline_allowed=True)
        budget = target.budgets[0]
        assert budget.is_relative is True
        assert budget.relative_pct == float(raw.rstrip("%"))

    def test_non_finite_percentage_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="finite percentage"):
            parse_slo_target("chain.total=+inf%", baseline_allowed=True)

    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            ("", "empty"),
            ("total=500,,ttfb=200", "Empty item"),
            ("total=500=100", "exactly one '='"),
            ("chain.=500", "KEY must not be empty"),
            ("bogus=500", "Unknown SLO key"),
            ("total=500,total=600", "Duplicate SLO key"),
            ("total=fast", "not a number"),
            ("total=0", "positive finite number"),
        ],
    )
    def test_bad_specs_rejected(self, raw: str, match: str) -> None:
        with pytest.raises(SLOSpecError, match=match):
            parse_slo_target(raw, baseline_allowed=True)

    def test_bad_percentage_number_rejected(self) -> None:
        with pytest.raises(SLOSpecError, match="not a percentage"):
            parse_slo_target("chain.total=+fast%", baseline_allowed=True)

    def test_legacy_parse_slo_spec_still_returns_dict(self) -> None:
        assert parse_slo_spec("total=500") == {"total": 500.0}


class TestNormalizeUrl:
    """URL normalization used by fingerprints."""

    def test_lowercases_scheme_and_host(self) -> None:
        assert normalize_url("HTTPS://Example.COM") == "https://example.com/"

    def test_drops_default_ports(self) -> None:
        assert normalize_url("https://example.com:443/x") == "https://example.com/x"
        assert normalize_url("http://example.com:80/x") == "http://example.com/x"

    def test_keeps_non_default_port(self) -> None:
        assert normalize_url("https://example.com:8443/x") == "https://example.com:8443/x"

    def test_drops_userinfo(self) -> None:
        assert normalize_url("https://user:pass@example.com/x") == "https://example.com/x"

    def test_empty_path_becomes_root(self) -> None:
        assert normalize_url("https://example.com") == "https://example.com/"

    def test_query_preserved_verbatim(self) -> None:
        assert normalize_url("https://example.com/x?b=2&a=1") == "https://example.com/x?b=2&a=1"


class TestLoadBaseline:
    """Baseline reports are validated on load."""

    def test_loads_valid_report(self, tmp_path: Path) -> None:
        report = _write_report(tmp_path / "base.json", CHAIN_HOPS)
        run = load_baseline(report)
        assert len(run.steps) == 2
        assert run.steps[0].values["total"] == 200.0
        assert run.steps[1].method == "GET"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SLOBaselineError, match="Cannot read"):
            load_baseline(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="not valid JSON"):
            load_baseline(path)

    def test_missing_steps_key(self, tmp_path: Path) -> None:
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="not an httptap report"):
            load_baseline(path)

    def test_error_step_rejected(self, tmp_path: Path) -> None:
        payload = _baseline_payload(CHAIN_HOPS)
        payload["steps"][1]["error"] = "connection refused"
        path = tmp_path / "errored.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="failed"):
            load_baseline(path)

    def test_missing_timing_field_rejected(self, tmp_path: Path) -> None:
        payload = _baseline_payload(CHAIN_HOPS)
        del payload["steps"][0]["timing"]["total_ms"]
        path = tmp_path / "notiming.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="total_ms"):
            load_baseline(path)

    def test_empty_steps_list_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"steps": []}), encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="no steps"):
            load_baseline(path)

    def test_non_object_step_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "badstep.json"
        path.write_text(json.dumps({"steps": [42]}), encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="malformed"):
            load_baseline(path)

    def test_missing_data_section_rejected(self, tmp_path: Path) -> None:
        payload = _baseline_payload(CHAIN_HOPS)
        del payload["steps"][0]["timing"]
        path = tmp_path / "missing.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="missing 'timing'"):
            load_baseline(path)

    def test_null_status_rejected(self, tmp_path: Path) -> None:
        payload = _baseline_payload(CHAIN_HOPS)
        payload["steps"][0]["response"]["status"] = None
        path = tmp_path / "nostatus.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SLOBaselineError, match="no response status"):
            load_baseline(path)


class TestFingerprintAlignment:
    """Mismatches name the first differing field."""

    def _steps(self, **overrides: object) -> list[StepMetrics]:
        defaults = [
            {"url": "https://hop.test/start", "step_number": 1, "status": 301, "total_ms": 400.0, "method": "GET"},
            {"url": "https://hop.test/final", "step_number": 2, "status": 200, "total_ms": 50.0, "method": "GET"},
        ]
        index = overrides.pop("index", 0)
        defaults[index].update(overrides)  # type: ignore[arg-type]
        return [_chain_step(**spec) for spec in defaults]  # type: ignore[arg-type]

    def _run(self, tmp_path: Path, hops: list[tuple[str, int, str, float]]) -> object:
        return load_baseline(_write_report(tmp_path / "base.json", hops))

    def test_matching_chain_passes(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, CHAIN_HOPS)
        ensure_baseline_matches(self._steps(), run)  # type: ignore[arg-type]

    def test_method_mismatch(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, CHAIN_HOPS)
        with pytest.raises(SLOBaselineError, match="methods differ"):
            ensure_baseline_matches(self._steps(method="POST"), run)  # type: ignore[arg-type]

    def test_source_mismatch(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, CHAIN_HOPS)
        with pytest.raises(SLOBaselineError, match="normalized sources differ"):
            ensure_baseline_matches(self._steps(url="https://other.test/start"), run)  # type: ignore[arg-type]

    def test_status_mismatch(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, CHAIN_HOPS)
        with pytest.raises(SLOBaselineError, match="status sequence differs"):
            ensure_baseline_matches(self._steps(status=302), run)  # type: ignore[arg-type]

    def test_final_target_mismatch(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, CHAIN_HOPS)
        with pytest.raises(SLOBaselineError, match="final target differ"):
            ensure_baseline_matches(self._steps(index=1, url="https://hop.test/other"), run)  # type: ignore[arg-type]

    def test_length_mismatch(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, CHAIN_HOPS)
        one_step = [_chain_step(url="https://hop.test/start", step_number=1, status=301, total_ms=400.0)]
        with pytest.raises(SLOBaselineError, match="chain length"):
            ensure_baseline_matches(one_step, run)  # type: ignore[arg-type]


class TestEvaluateTargetScopes:
    """Scope semantics and auditable step records."""

    def _redirect_chain(self, *, first_total: float = 400.0, final_total: float = 50.0) -> list[StepMetrics]:
        return [
            _chain_step(url="https://hop.test/start", step_number=1, status=301, total_ms=first_total),
            _chain_step(url="https://hop.test/final", step_number=2, status=200, total_ms=final_total),
        ]

    def test_final_ignores_slow_first_hop(self) -> None:
        target = parse_slo_target("final.total=100")
        result = evaluate_target(self._redirect_chain(), target)
        assert result.passed is True
        assert result.scope is SLOScope.FINAL
        assert result.checks[0].current_ms == 50.0
        assert result.checks[0].steps == (2,)

    def test_chain_sums_hops_and_fails(self) -> None:
        target = parse_slo_target("chain.total=300")
        result = evaluate_target(self._redirect_chain(), target)
        assert result.passed is False
        check = result.checks[0]
        assert check.scope is SLOScope.CHAIN
        assert check.current_ms == pytest.approx(450.0)
        assert check.threshold_ms == 300.0
        assert check.steps == (1, 2)
        assert check.aggregation == "sum"
        assert [sv.step for sv in check.step_values] == [1, 2]
        assert [sv.current_ms for sv in check.step_values] == [400.0, 50.0]
        assert result.violations[0].steps == (1, 2)

    def test_chain_passes_when_budget_covers_sum(self) -> None:
        target = parse_slo_target("chain.total=500")
        result = evaluate_target(self._redirect_chain(), target)
        assert result.passed is True

    def test_each_retains_violating_hops(self) -> None:
        target = parse_slo_target("each.total=100")
        result = evaluate_target(self._redirect_chain(), target)
        assert result.passed is False
        check = result.checks[0]
        assert check.scope is SLOScope.EACH
        assert check.steps == (1,)  # only the slow hop
        assert check.current_ms == 400.0
        assert len(check.step_values) == 2
        assert result.violations[0].steps == (1,)

    def test_each_pass_has_no_violations(self) -> None:
        target = parse_slo_target("each.total=500")
        result = evaluate_target(self._redirect_chain(), target)
        assert result.passed is True
        assert result.checks[0].steps == ()


class TestRelativeBaselineBudgets:
    """+20% budget boundary behavior against a fixed chain baseline."""

    def _baseline(self, tmp_path: Path) -> object:
        return load_baseline(_write_report(tmp_path / "base.json", CHAIN_HOPS))

    def _steps(self, totals: tuple[float, float]) -> list[StepMetrics]:
        return [
            _chain_step(url=url, step_number=i + 1, status=status, method=method, total_ms=total)
            for i, ((url, status, method, _base), total) in enumerate(zip(CHAIN_HOPS, totals, strict=True))
        ]

    def test_plus_19_percent_passes(self, tmp_path: Path) -> None:
        run = self._baseline(tmp_path)
        # Baseline chain total is 450ms; +19% is 535.5ms, limit 540ms.
        target = parse_slo_target("chain.total=+20%", baseline_allowed=True)
        result = evaluate_target(self._steps((238.0, 297.5)), target, baseline=run)  # type: ignore[arg-type]

        assert result.passed is True
        check = result.checks[0]
        assert check.baseline_ms == pytest.approx(450.0)
        assert check.current_ms == pytest.approx(535.5)
        assert check.threshold_ms == pytest.approx(540.0)
        assert check.relative_pct == pytest.approx(19.0)
        assert check.steps == (1, 2)
        assert check.budget_pct == 20.0

    def test_plus_21_percent_fails(self, tmp_path: Path) -> None:
        run = self._baseline(tmp_path)
        target = parse_slo_target("chain.total=+20%", baseline_allowed=True)
        result = evaluate_target(self._steps((242.0, 302.5)), target, baseline=run)  # type: ignore[arg-type]

        assert result.passed is False
        check = result.checks[0]
        assert check.baseline_ms == pytest.approx(450.0)
        assert check.current_ms == pytest.approx(544.5)
        assert check.relative_pct == pytest.approx(21.0)
        assert check.threshold_ms == pytest.approx(540.0)
        assert check.steps == (1, 2)
        violation = result.violations[0]
        assert violation.baseline_ms == pytest.approx(450.0)
        assert violation.relative_pct == pytest.approx(21.0)
        assert violation.delta_baseline_ms == pytest.approx(94.5)

    def test_mismatch_during_evaluation_raises_baseline_error(self, tmp_path: Path) -> None:
        run = self._baseline(tmp_path)
        target = parse_slo_target("chain.total=+20%", baseline_allowed=True)
        steps = self._steps((238.0, 297.5))
        steps[0] = _chain_step(url="https://different.test/start", step_number=1, status=301, total_ms=238.0)
        with pytest.raises(SLOBaselineError, match="normalized sources differ"):
            evaluate_target(steps, target, baseline=run)  # type: ignore[arg-type]

    def test_zero_baseline_with_positive_current_fails_and_pct_undefined(
        self,
        tmp_path: Path,
    ) -> None:
        zero_hops = [
            ("https://hop.test/start", 301, "GET", 0.0),
            ("https://hop.test/final", 200, "GET", 0.0),
        ]
        run = load_baseline(_write_report(tmp_path / "zero.json", zero_hops))
        target = parse_slo_target("chain.total=+20%", baseline_allowed=True)
        steps = [
            _chain_step(url="https://hop.test/start", step_number=1, status=301, total_ms=10.0),
            _chain_step(url="https://hop.test/final", step_number=2, status=200, total_ms=0.0),
        ]

        result = evaluate_target(steps, target, baseline=run)  # type: ignore[arg-type]

        assert result.passed is False
        check = result.checks[0]
        assert check.baseline_ms == 0.0
        assert check.current_ms == 10.0
        assert check.relative_pct is None
        assert check.threshold_ms == 0.0

    def test_relative_budget_without_baseline_run_raises_spec_error(self) -> None:
        target = SLOTarget(
            scope=SLOScope.CHAIN,
            budgets=(SLOBudget(key="total", relative_pct=20.0),),
        )
        steps = [_chain_step(url="https://hop.test/start", step_number=1, status=200, total_ms=10.0)]

        with pytest.raises(SLOSpecError, match="require --slo-baseline"):
            evaluate_target(steps, target)

    def test_describe_and_delta_helpers(self, tmp_path: Path) -> None:
        assert SLOBudget(key="total", relative_pct=-5.0).describe() == "total=-5%"
        assert SLOBudget(key="ttfb", absolute_ms=200.0).describe() == "ttfb=200ms"

        zero_hops = [
            ("https://hop.test/start", 301, "GET", 0.0),
            ("https://hop.test/final", 200, "GET", 0.0),
        ]
        run = load_baseline(_write_report(tmp_path / "zero.json", zero_hops))
        target = parse_slo_target("chain.total=+20%", baseline_allowed=True)
        steps = [
            _chain_step(url="https://hop.test/start", step_number=1, status=301, total_ms=10.0),
            _chain_step(url="https://hop.test/final", step_number=2, status=200, total_ms=0.0),
        ]

        result = evaluate_target(steps, target, baseline=run)  # type: ignore[arg-type]
        check = result.checks[0]
        assert check.delta_ms == 10.0
        assert check.delta_baseline_ms == 10.0

        assert SLOViolation(key="total", threshold_ms=10.0, actual_ms=10.0).delta_baseline_ms is None
        assert result.violations[0].current_ms == result.violations[0].actual_ms

    @pytest.mark.parametrize(
        ("budget", "match"),
        [
            (SLOBudget(key="total", relative_pct=20.0), "without an aligned baseline"),
            (SLOBudget(key="total"), "neither an absolute nor a relative"),
        ],
    )
    def test_resolve_threshold_invalid_budgets(self, budget: SLOBudget, match: str) -> None:
        from httptap.slo import _resolve_threshold

        with pytest.raises(SLOSpecError, match=match):
            _resolve_threshold(budget, None)

    def test_final_scope_with_all_error_steps_has_no_checks(self) -> None:
        target = parse_slo_target("final.total=100")
        bad_step = _step(step_number=1, error="boom")

        result = evaluate_target([bad_step], target)

        assert result.checks == ()
        assert result.passed is True
