"""Service Level Objective (SLO) evaluation for HTTP request timings.

This module implements SLO threshold checking: the user supplies
per-phase latency budgets and httptap evaluates the actual
measurements against them. A violation produces a non-zero exit code,
structured violation records in the JSON export, and human-readable
output for terminal and scripting pipelines.

SLO keys map one-to-one to ``TimingMetrics`` fields that express a
duration in milliseconds. The ``is_estimated`` flag is intentionally
excluded because it is boolean, not a duration.

Explicit target model
---------------------

An :class:`SLOTarget` pairs a :class:`SLOScope` with a tuple of
:class:`SLOBudget` objects. Three scopes are supported:

``final`` (default)
    Budgets apply to the *final* successful step of the chain — the
    response that actually served the request. This is the historical
    behavior and remains the default when no scope is given.

``chain``
    Each phase is aggregated across every successful hop using an
    auditable rule recorded per check (see
    :data:`CHAIN_AGGREGATION_RULES`; every phase currently sums).
    A ``chain.total`` budget therefore bounds the *whole* redirect
    chain rather than its last hop.

``each``
    Every hop is checked individually against the budgets. A check
    retains the concrete violating hops instead of a single aggregate.

Budgets are either absolute milliseconds (``total=300``) or a
*relative* percentage of a baseline value (``total=+20%``). Relative
budgets require ``--slo-baseline``: the baseline is an httptap JSON
report whose request fingerprint (methods, normalized sources, status
sequence, and final target) must match the current request before any
comparison is made.

Typical usage:
    >>> target = parse_slo_target("chain.total=300")
    >>> result = evaluate_target(steps, target)
    >>> if not result.passed:
    ...     for v in result.violations:
    ...         print(f"{v.key}: {v.actual_ms}ms > {v.threshold_ms}ms")

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .models import StepMetrics


# Keys accepted in ``--slo``. Each maps to a ``TimingMetrics.*_ms``
# field; adding a new entry here requires updating the ``timing_map``
# inside :func:`evaluate_slo` so the key-to-attribute mapping stays
# in sync.
SLO_KEYS: frozenset[str] = frozenset({"dns", "connect", "tls", "ttfb", "wait", "xfer", "total"})


class SLOScope(str, Enum):
    """Scope an :class:`SLOTarget` applies to.

    The enum inherits from :class:`str` so ``scope.value`` can be
    embedded directly in JSON and compared with plain strings.

    Attributes:
        FINAL: Final successful step only (default, legacy behavior).
        CHAIN: Per-phase aggregation across all successful hops.
        EACH: Every hop checked individually.

    """

    FINAL = "final"
    CHAIN = "chain"
    EACH = "each"


# Aggregation rule applied by the ``chain`` scope for each phase.
# Every timing phase is a duration spent *on that hop* (DNS lookup,
# handshake, body transfer, ...), so the chain-wide value is the sum
# across successful hops. The rule used is echoed on every
# :class:`SLOCheck` (``aggregation`` field) so exported results stay
# auditable without re-reading this source file.
CHAIN_AGGREGATION_RULES: Mapping[str, str] = dict.fromkeys(sorted(SLO_KEYS), "sum")


class SLOSpecError(ValueError):
    """Raised when an ``--slo`` specification cannot be parsed."""


class SLOBaselineError(SLOSpecError):
    """Raised when an SLO baseline is unusable.

    Covers unreadable files, non-JSON content, reports that are not
    httptap exports, reports containing failed steps, and fingerprint
    mismatches between the baseline and the current request. The CLI
    maps this to the exit code for a *usage* error (``EX_USAGE``),
    never to an SLO violation — callers can branch on this subtype.
    """


# ---------------------------------------------------------------------------
# Target model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SLOBudget:
    """One phase budget inside an :class:`SLOTarget`.

    Exactly one of ``absolute_ms`` / ``relative_pct`` is set.

    Attributes:
        key: Timing phase key (member of :data:`SLO_KEYS`).
        absolute_ms: Absolute budget in milliseconds.
        relative_pct: Signed relative budget as a percentage of the
            aligned baseline value (``20.0`` means ``+20%``).

    """

    key: str
    absolute_ms: float | None = None
    relative_pct: float | None = None

    @property
    def is_relative(self) -> bool:
        """Return ``True`` when the budget is a percentage of baseline."""
        return self.relative_pct is not None

    def describe(self) -> str:
        """Return the canonical text form used in panels and messages."""
        pct = self.relative_pct
        if pct is not None:
            sign = "+" if pct > 0 else ""
            return f"{self.key}={sign}{pct:g}%"
        return f"{self.key}={self.absolute_ms:g}ms"


@dataclass(frozen=True, slots=True)
class SLOTarget:
    """Explicit SLO target: a scope plus a set of phase budgets.

    Attributes:
        scope: Scope the budgets apply to.
        budgets: Phase budgets in user-supplied order.

    """

    scope: SLOScope = SLOScope.FINAL
    budgets: tuple[SLOBudget, ...] = ()

    @property
    def keys(self) -> tuple[str, ...]:
        """Budget keys in user-supplied order."""
        return tuple(budget.key for budget in self.budgets)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SLOStepValue:
    """One hop's measured value inside an :class:`SLOCheck`.

    Attributes:
        step: 1-indexed hop number.
        current_ms: Measured value on the current request, in ms.
        baseline_ms: Aligned baseline value, in ms, or ``None``.

    """

    step: int
    current_ms: float
    baseline_ms: float | None = None

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a JSON-ready dict (baseline only when present)."""
        data: dict[str, float | int] = {"step": self.step, "current_ms": self.current_ms}
        if self.baseline_ms is not None:
            data["baseline_ms"] = self.baseline_ms
        return data


@dataclass(frozen=True, slots=True)
class SLOCheck:
    """Auditable evaluation of one phase key for a target.

    Every budgeted key produces exactly one check regardless of scope.
    Scope-specific semantics:

    * ``final`` — :attr:`current_ms` is the final step's value,
      :attr:`steps` lists that step.
    * ``chain`` — :attr:`current_ms` is the aggregate (see
      :attr:`aggregation`) and :attr:`step_values` records each hop's
      contribution.
    * ``each`` — :attr:`current_ms` is the worst (maximum) value
      across hops; :attr:`steps` lists only the violating hops.

    Attributes:
        key: Timing phase key.
        passed: Whether the check met its budget.
        scope: Scope the check was evaluated under.
        threshold_ms: Effective budget in milliseconds (resolved
            against the baseline for relative budgets).
        current_ms: Measured value used for the comparison.
        steps: Relevant hop numbers (target hop for ``final``,
            contributors for ``chain``, violating hops for ``each``).
        step_values: Per-hop measured values.
        baseline_ms: Aligned baseline value, or ``None``.
        relative_pct: Measured difference vs baseline in percent
            (``(current - baseline) / baseline * 100``); ``None``
            without a baseline or when the baseline value is zero.
        budget_pct: User-supplied relative budget percentage when
            the budget is relative, else ``None``.
        aggregation: Aggregation rule used (``chain`` scope).

    """

    key: str
    passed: bool
    scope: SLOScope
    threshold_ms: float
    current_ms: float
    steps: tuple[int, ...]
    step_values: tuple[SLOStepValue, ...] = ()
    baseline_ms: float | None = None
    relative_pct: float | None = None
    budget_pct: float | None = None
    aggregation: str | None = None

    @property
    def delta_ms(self) -> float:
        """Difference between the measured value and the effective budget."""
        return self.current_ms - self.threshold_ms

    @property
    def delta_baseline_ms(self) -> float | None:
        """Absolute difference vs the baseline, in milliseconds."""
        if self.baseline_ms is None:
            return None
        return self.current_ms - self.baseline_ms

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a stable JSON-ready dict."""
        data: dict[str, Any] = {
            "key": self.key,
            "pass": self.passed,
            "scope": self.scope.value,
            "threshold_ms": self.threshold_ms,
            "current_ms": self.current_ms,
            "delta_ms": self.delta_ms,
            "steps": list(self.steps),
            "step_values": [sv.to_dict() for sv in self.step_values],
        }
        if self.aggregation is not None:
            data["aggregation"] = self.aggregation
        if self.budget_pct is not None:
            data["budget_pct"] = self.budget_pct
        if self.baseline_ms is not None:
            data["baseline_ms"] = self.baseline_ms
            data["delta_baseline_ms"] = self.delta_baseline_ms
            data["relative_pct"] = self.relative_pct
        return data


@dataclass(frozen=True, slots=True)
class SLOViolation:
    """A single threshold violation.

    Historically a violation compared one step's timing to one
    threshold. The same four JSON fields (``key``,
    ``threshold_ms``, ``actual_ms``, ``delta_ms``) are always emitted;
    the optional audit fields below are added only when a non-default
    scope or a baseline is used, keeping legacy JSON byte-identical.

    Attributes:
        key: Timing phase key (e.g., ``"total"``, ``"ttfb"``).
        threshold_ms: Effective budget, in milliseconds.
        actual_ms: Measured timing value, in milliseconds.
        scope: Scope value (``"final"`` by default).
        baseline_ms: Aligned baseline value, in ms, or ``None``.
        relative_pct: Measured difference vs baseline in percent.
        steps: Relevant hop numbers.

    """

    key: str
    threshold_ms: float
    actual_ms: float
    scope: str = SLOScope.FINAL.value
    baseline_ms: float | None = None
    relative_pct: float | None = None
    steps: tuple[int, ...] = ()

    @property
    def delta_ms(self) -> float:
        """Overrun over the budget, in milliseconds.

        By construction :func:`evaluate_slo` /
        :func:`evaluate_target` only emit violations where
        ``actual_ms > threshold_ms``, so this value is always strictly
        positive for objects produced by this module.
        """
        return self.actual_ms - self.threshold_ms

    @property
    def current_ms(self) -> float:
        """Alias for :attr:`actual_ms` (the measured current value)."""
        return self.actual_ms

    @property
    def delta_baseline_ms(self) -> float | None:
        """Absolute difference vs the baseline, in milliseconds."""
        if self.baseline_ms is None:
            return None
        return self.actual_ms - self.baseline_ms

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain ``dict`` for JSON export.

        Returns:
            The legacy four keys (``key``, ``threshold_ms``,
            ``actual_ms``, ``delta_ms``) plus audit fields
            (``scope``, ``steps``, and when a baseline is present
            ``baseline_ms``, ``delta_baseline_ms``, ``relative_pct``)
            only when a non-default scope or baseline is involved.

        """
        data: dict[str, Any] = {
            "key": self.key,
            "threshold_ms": self.threshold_ms,
            "actual_ms": self.actual_ms,
            "delta_ms": self.delta_ms,
        }
        enriched = self.scope != SLOScope.FINAL.value or self.baseline_ms is not None
        if enriched:
            data["scope"] = self.scope
            data["steps"] = list(self.steps)
        if self.baseline_ms is not None:
            data["baseline_ms"] = self.baseline_ms
            data["delta_baseline_ms"] = self.delta_baseline_ms
            data["relative_pct"] = self.relative_pct
        return data


@dataclass(frozen=True, slots=True)
class SLOResult:
    """Result of evaluating timings against an SLO target.

    The historical contract is preserved: for a default (``final``,
    absolute, no baseline) evaluation :meth:`to_dict` emits exactly
    ``pass`` / ``thresholds_ms`` / ``violations`` with violations
    sorted alphabetically by key. New fields appear only when new
    features (non-default scope or baseline) are used.

    Attributes:
        thresholds_ms: Effective budgets (resolved against baseline)
            keyed by SLO key.
        violations: Violations sorted alphabetically by key. When
            empty, the evaluation passed.
        scope: Scope the target was evaluated under.
        checks: One auditable record per budgeted key, sorted by key.
            Empty on the legacy path.
        baseline_path: Path of the baseline report used, or ``None``.

    The dataclass is ``frozen=True`` (shallow immutability) and is
    not hashable because :attr:`thresholds_ms` is a ``dict``.

    """

    thresholds_ms: dict[str, float] = field(default_factory=dict)
    violations: tuple[SLOViolation, ...] = ()
    scope: SLOScope = SLOScope.FINAL
    checks: tuple[SLOCheck, ...] = ()
    baseline_path: str | None = None

    @property
    def passed(self) -> bool:
        """Return ``True`` when every budget was met or undercut."""
        return len(self.violations) == 0

    @property
    def is_legacy(self) -> bool:
        """Return ``True`` for the historical final/absolute/no-baseline path."""
        return self.scope is SLOScope.FINAL and self.baseline_path is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain ``dict`` for JSON export.

        Returns:
            Legacy mapping (``pass``, ``thresholds_ms``,
            ``violations``) on the default path; the same mapping
            with ``scope`` inserted after ``pass`` and a ``checks``
            audit list when a non-default scope or baseline is used.

        """
        data: dict[str, Any] = {"pass": self.passed}
        if not self.is_legacy:
            data["scope"] = self.scope.value
        data["thresholds_ms"] = {key: self.thresholds_ms[key] for key in sorted(self.thresholds_ms)}
        if not self.is_legacy:
            data["checks"] = [check.to_dict() for check in self.checks]
        data["violations"] = [violation.to_dict() for violation in self.violations]
        return data


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_slo_spec(raw: str) -> dict[str, float]:
    """Parse a legacy absolute ``--slo`` specification string.

    The expected grammar is a comma-separated list of ``KEY=MS`` pairs.
    Whitespace around keys and values is tolerated. Keys are
    case-insensitive and normalised to lowercase.

    Args:
        raw: String passed via ``--slo``, e.g. ``"total=500,ttfb=200"``.

    Returns:
        Mapping of lowercase SLO key to threshold in milliseconds.

    Raises:
        SLOSpecError: If the string is empty, a pair is malformed, a
            key is unknown, a key is duplicated, or a value is not a
            positive, finite number.

    Examples:
        >>> parse_slo_spec("total=500,ttfb=200")
        {'total': 500.0, 'ttfb': 200.0}
        >>> parse_slo_spec("Total=500")
        {'total': 500.0}

    """
    if not raw or not raw.strip():
        msg = "SLO specification is empty (expected KEY=MS[,KEY=MS...])."
        raise SLOSpecError(msg)

    thresholds: dict[str, float] = {}

    for pair in raw.split(","):
        token = pair.strip()
        if not token:
            msg = f"Empty item in SLO specification '{raw}' (expected KEY=MS[,KEY=MS...])."
            raise SLOSpecError(msg)

        if token.count("=") != 1:
            msg = f"Invalid SLO item '{token}' (expected exactly one '=' between KEY and MS)."
            raise SLOSpecError(msg)

        key_raw, value_raw = token.split("=", 1)
        key = key_raw.strip().lower()
        value_str = value_raw.strip()

        if not key:
            msg = f"Invalid SLO item '{token}' (KEY must not be empty)."
            raise SLOSpecError(msg)

        if key not in SLO_KEYS:
            allowed = ", ".join(sorted(SLO_KEYS))
            msg = f"Unknown SLO key '{key}'. Valid keys: {allowed}."
            raise SLOSpecError(msg)

        if key in thresholds:
            msg = f"Duplicate SLO key '{key}' in specification '{raw}'."
            raise SLOSpecError(msg)

        try:
            value = float(value_str)
        except ValueError as exc:
            msg = f"Invalid SLO value for '{key}': '{value_str}' is not a number."
            raise SLOSpecError(msg) from exc

        if not math.isfinite(value) or value <= 0:
            msg = f"Invalid SLO value for '{key}': '{value_str}' must be a positive finite number of milliseconds."
            raise SLOSpecError(msg)

        thresholds[key] = value

    return thresholds


def parse_slo_target(
    raw: str,
    *,
    baseline_allowed: bool = False,
) -> SLOTarget:
    """Parse an ``--slo`` specification into an explicit :class:`SLOTarget`.

    Grammar (commas separate items, whitespace tolerated, keys
    case-insensitive)::

        [SCOPE.]KEY=VALUE[, [SCOPE.]KEY=VALUE]*

    * ``SCOPE`` is ``final`` (default), ``chain``, or ``each``. Either
      every item carries a scope prefix or none do; mixing is rejected
      so one specification evaluates to one auditable result.
    * ``VALUE`` is a positive finite number of milliseconds
      (``total=300``), or — only when a baseline is available — a
      signed percentage (``total=+20%``, ``ttfb=-5%``, ``total=20%``).

    Args:
        raw: String passed via ``--slo``.
        baseline_allowed: Whether ``--slo-baseline`` was supplied;
            gates percentage values.

    Returns:
        Parsed target.

    Raises:
        SLOSpecError: On empty input, malformed items, unknown keys
            or scopes, duplicated keys, mixed prefixes, bad numbers,
            or percentage values without a baseline.

    Examples:
        >>> parse_slo_target("total=500").scope.value
        'final'
        >>> target = parse_slo_target("chain.total=300")
        >>> target.scope.value
        'chain'

    """
    if not raw or not raw.strip():
        msg = "SLO specification is empty (expected KEY=MS[,KEY=MS...])."
        raise SLOSpecError(msg)

    scope_values = {entry.value for entry in SLOScope}
    budgets: list[SLOBudget] = []
    scope: SLOScope | None = None
    seen: set[str] = set()

    for pair in raw.split(","):
        budget, item_scope = _parse_target_item(
            pair.strip(),
            raw,
            scope_values=scope_values,
            baseline_allowed=baseline_allowed,
        )

        if item_scope is not None:
            scope = _consistent_scope(scope, item_scope, raw)
        elif scope is not None:
            msg = f"Mix of scoped and unscoped SLO items in '{raw}'; use one form consistently."
            raise SLOSpecError(msg)

        if budget.key in seen:
            msg = f"Duplicate SLO key '{budget.key}' in specification '{raw}'."
            raise SLOSpecError(msg)
        seen.add(budget.key)

        budgets.append(budget)

    return SLOTarget(scope=scope or SLOScope.FINAL, budgets=tuple(budgets))


def _parse_target_item(
    token: str,
    raw: str,
    *,
    scope_values: set[str],
    baseline_allowed: bool,
) -> tuple[SLOBudget, SLOScope | None]:
    """Parse one ``[scope.]key=value`` item into a budget and optional scope."""
    if not token:
        msg = f"Empty item in SLO specification '{raw}' (expected KEY=MS[,KEY=MS...])."
        raise SLOSpecError(msg)

    if token.count("=") != 1:
        msg = f"Invalid SLO item '{token}' (expected exactly one '=' between KEY and VALUE)."
        raise SLOSpecError(msg)

    key_raw, value_raw = token.split("=", 1)
    key_token = key_raw.strip().lower()
    value_str = value_raw.strip()

    item_scope: SLOScope | None = None
    if "." in key_token:
        prefix, rest = key_token.split(".", 1)
        if prefix not in scope_values:
            allowed = ", ".join(sorted(scope_values))
            msg = f"Unknown SLO scope '{prefix}'. Valid scopes: {allowed}."
            raise SLOSpecError(msg)
        item_scope = SLOScope(prefix)
        key_token = rest

    if not key_token:
        msg = f"Invalid SLO item '{token}' (KEY must not be empty)."
        raise SLOSpecError(msg)

    if key_token not in SLO_KEYS:
        allowed = ", ".join(sorted(SLO_KEYS))
        msg = f"Unknown SLO key '{key_token}'. Valid keys: {allowed}."
        raise SLOSpecError(msg)

    budget = _parse_budget(key_token, value_str, baseline_allowed=baseline_allowed)
    return budget, item_scope


def _consistent_scope(
    scope: SLOScope | None,
    item_scope: SLOScope,
    raw: str,
) -> SLOScope:
    """Ensure every prefixed item shares one scope."""
    if scope is None:
        return item_scope
    if scope is not item_scope:
        msg = (
            f"Mixed SLO scopes in '{raw}': all items must share one scope "
            "(drop prefixes for the default 'final' scope)."
        )
        raise SLOSpecError(msg)
    return scope


def _parse_budget(key: str, value_str: str, *, baseline_allowed: bool) -> SLOBudget:
    """Parse one budget value (absolute ms or signed percentage)."""
    if value_str.endswith("%"):
        if not baseline_allowed:
            msg = f"Relative SLO budget '{key}={value_str}' requires --slo-baseline pointing at an httptap report."
            raise SLOSpecError(msg)
        pct_str = value_str[:-1].strip()
        try:
            pct = float(pct_str)
        except ValueError as exc:
            msg = f"Invalid SLO value for '{key}': '{value_str}' is not a percentage."
            raise SLOSpecError(msg) from exc
        if not math.isfinite(pct):
            msg = f"Invalid SLO value for '{key}': '{value_str}' must be a finite percentage."
            raise SLOSpecError(msg)
        return SLOBudget(key=key, relative_pct=pct)

    try:
        value = float(value_str)
    except ValueError as exc:
        msg = f"Invalid SLO value for '{key}': '{value_str}' is not a number."
        raise SLOSpecError(msg) from exc

    if not math.isfinite(value) or value <= 0:
        msg = f"Invalid SLO value for '{key}': '{value_str}' must be a positive finite number of milliseconds."
        raise SLOSpecError(msg)

    return SLOBudget(key=key, absolute_ms=value)


# ---------------------------------------------------------------------------
# Baseline reports and fingerprints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineStep:
    """One hop read from a baseline report.

    Attributes:
        url: Request URL exactly as stored in the report.
        method: Upper-cased request method.
        status: Response status code.
        values: Timing values keyed by :data:`SLO_KEYS`, in ms.

    """

    url: str
    method: str
    status: int
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BaselineRun:
    """A loaded baseline report.

    Attributes:
        path: Path the report was read from.
        steps: Successful hops recorded in the report.

    """

    path: str
    steps: tuple[BaselineStep, ...]


def load_baseline(path: str | Path) -> BaselineRun:
    """Load an httptap JSON report for use as an SLO baseline.

    Only the report shape written by :class:`~httptap.exporter.JSONExporter`
    is accepted: a top-level ``steps`` list whose entries carry ``url``,
    ``request.method``, ``response.status`` and ``timing.*_ms``.

    Args:
        path: Path to the baseline JSON report.

    Returns:
        Loaded :class:`BaselineRun`.

    Raises:
        SLOBaselineError: If the file cannot be read, is not JSON, is
            not an httptap report, contains malformed or failed steps,
            or omits a required timing field.

    """
    report_path = Path(path)
    payload = _read_baseline_payload(report_path, display=path)

    if not isinstance(payload, dict) or "steps" not in payload:
        msg = f"SLO baseline '{path}' is not an httptap report (missing top-level 'steps')."
        raise SLOBaselineError(msg)

    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        msg = f"SLO baseline '{path}' contains no steps to compare against."
        raise SLOBaselineError(msg)

    steps = tuple(
        _parse_baseline_step(raw_step, path=path, label=f"step {index + 1}") for index, raw_step in enumerate(raw_steps)
    )
    return BaselineRun(path=str(report_path), steps=steps)


def _read_baseline_payload(report_path: Path, *, display: object) -> object:
    """Read and JSON-decode a baseline file."""
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot read SLO baseline '{display}': {exc}."
        raise SLOBaselineError(msg) from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"SLO baseline '{display}' is not valid JSON: {exc}."
        raise SLOBaselineError(msg) from exc


def _parse_baseline_step(raw_step: object, *, path: object, label: str) -> BaselineStep:
    """Parse one step object of a baseline report."""
    if not isinstance(raw_step, dict):
        msg = f"SLO baseline '{path}' {label} is malformed (expected an object)."
        raise SLOBaselineError(msg)
    if raw_step.get("error"):
        msg = f"SLO baseline '{path}' {label} failed ({raw_step['error']}); cannot use it as a budget."
        raise SLOBaselineError(msg)

    try:
        url = raw_step["url"]
        request = raw_step.get("request") or {}
        method = (request.get("method") or "GET").upper()
        response = raw_step["response"]
        status = response["status"]
        timing = raw_step["timing"]
    except (AttributeError, KeyError, TypeError) as exc:
        msg = f"SLO baseline '{path}' {label} is missing '{exc.args[0]}' data."
        raise SLOBaselineError(msg) from exc

    if status is None:
        msg = f"SLO baseline '{path}' {label} has no response status; cannot align the chain."
        raise SLOBaselineError(msg)

    values = _baseline_timing_values(timing, path=path, label=label)
    return BaselineStep(url=url, method=method, status=int(status), values=values)


def _baseline_timing_values(timing: object, *, path: object, label: str) -> dict[str, float]:
    """Read and convert every ``<key>_ms`` field of one baseline step."""
    timing_mapping = cast("dict[str, object]", timing)
    values: dict[str, float] = {}
    for key in SLO_KEYS:
        timing_key = f"{key}_ms"
        try:
            # JSON values are objects at type level; TypeError below covers
            # runtime values that do not convert.
            values[key] = float(timing_mapping[timing_key])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"SLO baseline '{path}' {label} is missing timing field '{timing_key}'."
            raise SLOBaselineError(msg) from exc
    return values


def normalize_url(url: str) -> str:
    """Normalize a request URL for fingerprint comparison.

    Normalization rules:

    * scheme and host lower-cased;
    * userinfo dropped;
    * default port (80/443) dropped;
    * empty path rendered as ``/``;
    * query string preserved verbatim.

    These rules keep two URLs that name the same hop identical while
    avoiding lossy transformations (the query is *not* sorted, the
    path is *not* case-folded).

    Args:
        url: Raw URL from a step or baseline report.

    Returns:
        Normalized URL.

    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    default_port = 443 if scheme == "https" else 80
    port = parts.port
    netloc = host if not port or port == default_port else f"{host}:{port}"
    path = parts.path or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{netloc}{path}{query}"


@dataclass(frozen=True, slots=True)
class RequestFingerprint:
    """Identity of a request chain used to align a baseline.

    Attributes:
        methods: Per-hop upper-cased request methods.
        sources: Per-hop normalized request URLs.
        statuses: Per-hop response status codes.
        final_target: Normalized URL of the final hop.

    """

    methods: tuple[str, ...]
    sources: tuple[str, ...]
    statuses: tuple[int, ...]
    final_target: str


def fingerprint_current(steps: Sequence[StepMetrics]) -> RequestFingerprint:
    """Build the fingerprint of the current successful request chain."""
    return RequestFingerprint(
        methods=tuple((s.request_method or "GET").upper() for s in steps),
        sources=tuple(normalize_url(s.url) for s in steps),
        statuses=tuple(int(s.response.status or 0) for s in steps),
        final_target=normalize_url(steps[-1].url),
    )


def fingerprint_baseline(run: BaselineRun) -> RequestFingerprint:
    """Build the fingerprint of a loaded baseline."""
    return RequestFingerprint(
        methods=tuple(step.method for step in run.steps),
        sources=tuple(normalize_url(step.url) for step in run.steps),
        statuses=tuple(step.status for step in run.steps),
        final_target=normalize_url(run.steps[-1].url),
    )


def ensure_baseline_matches(
    steps: Sequence[StepMetrics],
    run: BaselineRun,
) -> None:
    """Verify that a baseline describes the same request as ``steps``.

    Alignment is checked in a fixed order so the error always names
    the first differing field: chain length, methods, final target,
    normalized sources, then status sequence. The final target is
    checked before the source tuple on purpose: it is the user's
    destination and overlaps with the final entry of the source
    tuple, so checking it first keeps that mismatch identifiable
    (a changed final hop URL reports ``final target``, a changed
    intermediate hop URL reports ``normalized sources``).

    Args:
        steps: Current request chain (successful steps).
        run: Loaded baseline.

    Raises:
        SLOBaselineError: With a message naming the mismatching field
            and the values on both sides.

    """
    current = fingerprint_current(steps)
    baseline = fingerprint_baseline(run)
    baseline_path = run.path

    if len(baseline.statuses) != len(current.statuses):
        msg = (
            f"SLO baseline '{baseline_path}' does not match the current request: "
            f"chain length differs (baseline {len(baseline.statuses)} hops vs current {len(current.statuses)})."
        )
        raise SLOBaselineError(msg)

    if baseline.methods != current.methods:
        msg = (
            f"SLO baseline '{baseline_path}' does not match the current request: "
            f"methods differ (baseline {list(baseline.methods)} vs current {list(current.methods)})."
        )
        raise SLOBaselineError(msg)

    if baseline.final_target != current.final_target:
        msg = (
            f"SLO baseline '{baseline_path}' does not match the current request: "
            f"final target differs (baseline {baseline.final_target} vs current {current.final_target})."
        )
        raise SLOBaselineError(msg)

    if baseline.sources != current.sources:
        msg = (
            f"SLO baseline '{baseline_path}' does not match the current request: "
            f"normalized sources differ (baseline {list(baseline.sources)} vs current {list(current.sources)})."
        )
        raise SLOBaselineError(msg)

    if baseline.statuses != current.statuses:
        msg = (
            f"SLO baseline '{baseline_path}' does not match the current request: "
            f"status sequence differs (baseline {list(baseline.statuses)} vs current {list(current.statuses)})."
        )
        raise SLOBaselineError(msg)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _timing_value(step: StepMetrics, key: str) -> float:
    """Read one SLO-keyed timing value from a step."""
    return float(getattr(step.timing, f"{key}_ms"))


def _relative_pct(current: float, baseline: float) -> float | None:
    """Percent difference vs baseline; ``None`` when undefined (base 0)."""
    if baseline == 0:
        return 0.0 if current == 0 else None
    return (current - baseline) / baseline * 100.0


def _resolve_threshold(budget: SLOBudget, baseline: float | None) -> float:
    """Resolve the effective millisecond threshold of a budget."""
    if budget.is_relative:
        if baseline is None:
            msg = "Relative SLO budget used without an aligned baseline value."
            raise SLOSpecError(msg)
        return baseline * (1.0 + (budget.relative_pct or 0.0) / 100.0)
    if budget.absolute_ms is None:
        msg = f"SLO budget '{budget.key}' has neither an absolute nor a relative value."
        raise SLOSpecError(msg)
    return budget.absolute_ms


def evaluate_slo(
    step: StepMetrics,
    thresholds: Mapping[str, float],
) -> SLOResult:
    """Evaluate timings on a single step against absolute SLO thresholds.

    Args:
        step: Step whose ``timing`` is compared to the thresholds.
        thresholds: Mapping of SLO key to threshold in milliseconds.
            Every key must be a member of :data:`SLO_KEYS`.

    Returns:
        :class:`SLOResult` listing any violations in ascending
        alphabetical order of the user-supplied threshold keys. The
        order is stable so the JSON export is reproducible.

    Raises:
        SLOSpecError: If ``thresholds`` contains a key that is not a
            member of :data:`SLO_KEYS`.

    """
    unknown = set(thresholds) - SLO_KEYS
    if unknown:
        allowed = ", ".join(sorted(SLO_KEYS))
        bad = ", ".join(sorted(unknown))
        msg = f"Unknown SLO key(s): {bad}. Valid keys: {allowed}."
        raise SLOSpecError(msg)

    # Keep this mapping in lockstep with SLO_KEYS (the frozenset above).
    timing_map: dict[str, float] = {
        "dns": step.timing.dns_ms,
        "connect": step.timing.connect_ms,
        "tls": step.timing.tls_ms,
        "ttfb": step.timing.ttfb_ms,
        "wait": step.timing.wait_ms,
        "xfer": step.timing.xfer_ms,
        "total": step.timing.total_ms,
    }

    violations = tuple(
        SLOViolation(key=key, threshold_ms=thresholds[key], actual_ms=timing_map[key])
        for key in sorted(thresholds)
        if timing_map[key] > thresholds[key]
    )

    return SLOResult(thresholds_ms=dict(thresholds), violations=violations)


def evaluate_target(
    steps: Sequence[StepMetrics],
    target: SLOTarget,
    *,
    baseline: BaselineRun | None = None,
) -> SLOResult:
    """Evaluate a request chain against an explicit :class:`SLOTarget`.

    Args:
        steps: Current request chain.
        target: Parsed target (scope + budgets).
        baseline: Loaded baseline, required for relative budgets.

    Returns:
        :class:`SLOResult` with one :class:`SLOCheck` per budget key,
        sorted alphabetically, plus violations for the failing ones.

    Raises:
        SLOBaselineError: If ``baseline`` is supplied but its
            fingerprint does not match ``steps``.
        SLOSpecError: If a relative budget lacks a baseline.

    """
    if baseline is not None:
        ensure_baseline_matches([s for s in steps if not s.has_error], baseline)

    if any(budget.is_relative for budget in target.budgets) and baseline is None:
        msg = "Relative SLO budgets require --slo-baseline."
        raise SLOSpecError(msg)

    if target.scope is SLOScope.FINAL:
        checks = [
            _evaluate_final(step=step, budget=budget, baseline=baseline) for step, budget in _final_steps(steps, target)
        ]
    elif target.scope is SLOScope.CHAIN:
        checks = [_evaluate_chain(steps=steps, budget=budget, baseline=baseline) for budget in target.budgets]
    else:
        checks = [_evaluate_each(steps=steps, budget=budget, baseline=baseline) for budget in target.budgets]

    checks = sorted(checks, key=lambda check: check.key)
    return _build_result(checks, scope=target.scope, baseline=baseline)


def _final_steps(
    steps: Sequence[StepMetrics],
    target: SLOTarget,
) -> tuple[tuple[StepMetrics, SLOBudget], ...]:
    """Pair each budget with the final successful step."""
    step = select_step_for_evaluation(steps)
    if step is None:
        return ()
    return tuple((step, budget) for budget in target.budgets)


def _evaluate_final(
    *,
    step: StepMetrics,
    budget: SLOBudget,
    baseline: BaselineRun | None,
) -> SLOCheck:
    """Evaluate one budget against the final successful step."""
    current = _timing_value(step, budget.key)
    base = baseline.steps[-1].values[budget.key] if baseline is not None else None
    threshold = _resolve_threshold(budget, base)
    pct = _relative_pct(current, base) if base is not None else None
    return SLOCheck(
        key=budget.key,
        passed=current <= threshold,
        scope=SLOScope.FINAL,
        threshold_ms=threshold,
        current_ms=current,
        steps=(step.step_number,),
        step_values=(SLOStepValue(step=step.step_number, current_ms=current, baseline_ms=base),),
        baseline_ms=base,
        relative_pct=pct,
        budget_pct=budget.relative_pct,
    )


def _evaluate_chain(
    *,
    steps: Sequence[StepMetrics],
    budget: SLOBudget,
    baseline: BaselineRun | None,
) -> SLOCheck:
    """Evaluate one budget as a chain-wide aggregate."""
    successful = [step for step in steps if not step.has_error]
    rule = CHAIN_AGGREGATION_RULES[budget.key]
    current = sum(_timing_value(step, budget.key) for step in successful)
    base = sum(bstep.values[budget.key] for bstep in baseline.steps) if baseline is not None else None
    threshold = _resolve_threshold(budget, base)
    step_values = tuple(
        SLOStepValue(
            step=step.step_number,
            current_ms=_timing_value(step, budget.key),
            baseline_ms=baseline.steps[index].values[budget.key] if baseline is not None else None,
        )
        for index, step in enumerate(successful)
    )
    return SLOCheck(
        key=budget.key,
        passed=current <= threshold,
        scope=SLOScope.CHAIN,
        threshold_ms=threshold,
        current_ms=current,
        steps=tuple(step.step_number for step in successful),
        step_values=step_values,
        baseline_ms=base,
        relative_pct=_relative_pct(current, base) if base is not None else None,
        budget_pct=budget.relative_pct,
        aggregation=rule,
    )


def _evaluate_each(
    *,
    steps: Sequence[StepMetrics],
    budget: SLOBudget,
    baseline: BaselineRun | None,
) -> SLOCheck:
    """Evaluate one budget against every hop individually."""
    successful = [step for step in steps if not step.has_error]
    step_values: list[SLOStepValue] = []
    violating: list[int] = []

    for index, step in enumerate(successful):
        current = _timing_value(step, budget.key)
        base = baseline.steps[index].values[budget.key] if baseline is not None else None
        threshold = _resolve_threshold(budget, base)
        if current > threshold:
            violating.append(step.step_number)
        step_values.append(SLOStepValue(step=step.step_number, current_ms=current, baseline_ms=base))

    # The audited value is the worst hop; locate its aligned baseline.
    worst = max(step_values, key=lambda sv: sv.current_ms)
    current = worst.current_ms
    base = worst.baseline_ms
    threshold = _resolve_threshold(budget, base)
    return SLOCheck(
        key=budget.key,
        passed=not violating,
        scope=SLOScope.EACH,
        threshold_ms=threshold,
        current_ms=current,
        steps=tuple(violating),
        step_values=tuple(step_values),
        baseline_ms=base,
        relative_pct=_relative_pct(current, base) if base is not None else None,
        budget_pct=budget.relative_pct,
    )


def _build_result(
    checks: Sequence[SLOCheck],
    *,
    scope: SLOScope,
    baseline: BaselineRun | None,
) -> SLOResult:
    """Assemble a result from sorted checks."""
    thresholds = {check.key: check.threshold_ms for check in checks}
    violations = tuple(
        SLOViolation(
            key=check.key,
            threshold_ms=check.threshold_ms,
            actual_ms=check.current_ms,
            scope=scope.value,
            baseline_ms=check.baseline_ms,
            relative_pct=check.relative_pct,
            steps=check.steps,
        )
        for check in checks
        if not check.passed
    )
    return SLOResult(
        thresholds_ms=thresholds,
        violations=violations,
        scope=scope,
        checks=tuple(checks),
        baseline_path=baseline.path if baseline is not None else None,
    )


def select_step_for_evaluation(steps: Sequence[StepMetrics]) -> StepMetrics | None:
    """Pick the step whose timing should be checked against SLO.

    By convention, SLOs apply to the *final* successful step of a
    redirect chain (the one that actually served the user's request).
    If every step errored out, returns ``None`` — the caller is
    expected to treat that as a network failure rather than an SLO
    violation.

    Args:
        steps: Steps returned by ``HTTPTapAnalyzer.analyze_url``.

    Returns:
        Final successful step, or ``None`` if there is no such step.

    """
    for step in reversed(steps):
        if not step.has_error:
            return step
    return None
