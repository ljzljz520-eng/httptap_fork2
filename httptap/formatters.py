"""Output formatting utilities for HTTP request analysis.

This module provides formatters for converting metrics and data
into human-readable formats with Rich markup support.
"""

from collections.abc import Sequence

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from .constants import (
    BYTES_PER_KIB,
    CERT_EXPIRY_CRITICAL_DAYS,
    CERT_EXPIRY_WARNING_DAYS,
    HTTP_REDIRECT_MAX,
    HTTP_REDIRECT_MIN,
    HTTP_SUCCESS_MAX,
    HTTP_SUCCESS_MIN,
    PROXY_SOURCE_CLI,
    PROXY_SOURCE_DISABLED,
    PROXY_SOURCE_NO_MATCH,
    PROXY_SOURCE_NO_PROXY,
)
from .models import StepMetrics
from .slo import SLOCheck, SLOResult, SLOScope, SLOViolation


def format_step_header(step: StepMetrics) -> str:
    """Format step header with number and URL.

    Args:
        step: Step metrics to format.

    Returns:
        Formatted header string with rich markup.

    """
    return f"[bold cyan]Step {step.step_number}:[/bold cyan] [dim]{escape(step.url)}[/dim]"


def format_error(step: StepMetrics) -> Panel:
    """Format error information as a Rich panel.

    Args:
        step: Step metrics with error.

    Returns:
        Rich Panel with formatted error information.

    """
    error_text = Text()
    error_text.append("❌ ", style="bold red")
    error_text.append(str(step.error), style="red")

    if step.note:
        error_text.append("\n\n")
        error_text.append("💡 ", style="yellow")
        error_text.append(step.note, style="yellow")

    return Panel(
        error_text,
        title="[bold red]Error[/bold red]",
        border_style="red",
        padding=(0, 1),
    )


def _format_proxy_part(step: StepMetrics) -> str:
    """Format proxy information for display.

    Args:
        step: Step metrics with proxy data.

    Returns:
        Formatted proxy string with Rich markup.

    """
    source = step.network.proxy_source
    if step.proxied_via:
        hint = "from arg --proxy" if source == PROXY_SOURCE_CLI else f"from env {source}"
        return f"Proxy: {escape(step.proxied_via)} ({hint})"
    if source == PROXY_SOURCE_NO_PROXY:
        return "[yellow]Proxy: none (bypassed by env no_proxy)[/yellow]"
    if source == PROXY_SOURCE_DISABLED:
        return 'Proxy: disabled (from --proxy "")'
    if source == PROXY_SOURCE_NO_MATCH:
        return "Proxy: direct (no matching proxy scheme in env)"
    return "Proxy: direct"


def format_network_info(step: StepMetrics) -> str | None:
    """Format network and security information.

    Args:
        step: Step metrics with network data.

    Returns:
        Formatted network info string or None if no data.

    """
    parts = []

    ip = step.network.ip
    if ip:
        family = step.network.ip_family
        parts.append(f"IP: {escape(ip)} ({family})" if family else f"IP: {escape(ip)}")

    http_version = step.network.http_version
    if http_version:
        parts.append(f"HTTP: {escape(http_version)}")

    parts.append(_format_proxy_part(step))

    for label, value in (
        ("TLS", step.network.tls_version),
        ("Cipher", step.network.tls_cipher),
        ("Cert", step.network.cert_cn),
        ("Issuer", step.network.cert_issuer),
    ):
        if value:
            parts.append(f"{label}: {escape(value)}")

    days_left = step.network.cert_days_left
    if days_left is not None:
        days_style = (
            "red"
            if days_left < CERT_EXPIRY_CRITICAL_DAYS
            else "yellow"
            if days_left < CERT_EXPIRY_WARNING_DAYS
            else "green"
        )
        parts.append(f"[{days_style}]Expires: {days_left}d[/{days_style}]")
    if step.network.tls_verified is False:
        parts.append("[bold red]⚠ TLS verification disabled[/bold red]")
    elif step.network.tls_custom_ca:
        parts.append("TLS CA: custom bundle")

    return f"  [dim]{' | '.join(parts)}[/dim]" if parts else None


def format_response_info(step: StepMetrics) -> str | None:
    """Format HTTP response information.

    Args:
        step: Step metrics with response data.

    Returns:
        Formatted response info string or None if no data.

    """
    parts = []

    if step.response.status:
        status = step.response.status
        if HTTP_SUCCESS_MIN <= status <= HTTP_SUCCESS_MAX:
            status_style = "green"
        elif HTTP_REDIRECT_MIN <= status <= HTTP_REDIRECT_MAX:
            status_style = "yellow"
        else:
            status_style = "red"
        parts.append(f"[{status_style}]Status: {status}[/{status_style}]")

    if step.response.bytes > 0:
        size_str = format_bytes_human(step.response.bytes)
        parts.append(f"Size: {size_str}")

    if step.response.server:
        parts.append(f"Server: {escape(step.response.server)}")
    if step.response.location:
        parts.append(f"→ [cyan]{escape(step.response.location)}[/cyan]")

    return f"  {' | '.join(parts)}" if parts else None


def format_bytes_human(num_bytes: int) -> str:
    """Format bytes in human-readable format.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string with appropriate unit.

    Examples:
        >>> format_bytes_human(512)
        '512 B'
        >>> format_bytes_human(2048)
        '2.0 KB'
        >>> format_bytes_human(1048576)
        '1.0 MB'

    """
    size_kib = num_bytes / BYTES_PER_KIB
    if size_kib < 1:
        return f"{num_bytes} B"
    if size_kib < BYTES_PER_KIB:
        return f"{size_kib:.1f} KB"
    return f"{size_kib / BYTES_PER_KIB:.1f} MB"


def format_metrics_line(
    step: StepMetrics,
    *,
    slo_result: SLOResult | None = None,
) -> str:
    """Format step metrics as machine-readable line.

    Args:
        step: Step metrics to format.
        slo_result: Evaluated SLO result. When provided, appends
            ``slo=pass`` or ``slo=fail slo_violations=<keys>`` to the
            output so downstream tooling can branch on a single line.

    Returns:
        Space-separated key=value pairs.

    Examples:
        >>> format_metrics_line(step)
        'Step 1: dns=1.5 ... total=234.7 status=200 bytes=1256 proxy=direct'
        >>> format_metrics_line(step, slo_result)
        'Step 1: ... proxy=direct slo=fail slo_violations=total,ttfb'

    """
    parts = [
        f"dns={step.timing.dns_ms:.1f}",
        f"connect={step.timing.connect_ms:.1f}",
        f"tls={step.timing.tls_ms:.1f}",
        f"ttfb={step.timing.ttfb_ms:.1f}",
        f"total={step.timing.total_ms:.1f}",
        f"status={step.response.status}",
        f"bytes={step.response.bytes}",
    ]

    if step.network.ip:
        parts.append(f"ip={step.network.ip}")
    if step.network.ip_family:
        parts.append(f"family={step.network.ip_family}")
    if step.network.tls_version:
        parts.append(f"tls_version={step.network.tls_version}")

    parts.extend(_proxy_tokens(step))

    if slo_result is not None:
        if not slo_result.is_legacy:
            parts.extend(_enriched_slo_tokens(slo_result))

        if slo_result.passed:
            parts.append("slo=pass")
        else:
            violated = ",".join(v.key for v in slo_result.violations)
            parts.append("slo=fail")
            parts.append(f"slo_violations={violated}")

    return f"Step {step.step_number}: {' '.join(parts)}"


def _proxy_tokens(step: StepMetrics) -> list[str]:
    """Build the proxy/proxy_from token(s) for one step."""
    if step.proxied_via:
        src = step.network.proxy_source
        hint = "arg" if src == PROXY_SOURCE_CLI else f"env:{src}"
        return [f"proxy={step.proxied_via} proxy_from={hint}"]
    if step.network.proxy_source == PROXY_SOURCE_NO_PROXY:
        return ["proxy=none proxy_from=env:no_proxy"]
    if step.network.proxy_source == PROXY_SOURCE_DISABLED:
        return ['proxy=disabled proxy_from=--proxy ""']
    if step.network.proxy_source == PROXY_SOURCE_NO_MATCH:
        return ["proxy=direct proxy_from=no_scheme_match"]
    return ["proxy=direct"]


def _enriched_slo_tokens(result: SLOResult) -> list[str]:
    """Build scope/per-check tokens for an enriched (non-legacy) SLO result."""
    tokens = [f"slo_scope={result.scope.value}"]
    for check in result.checks:
        check_steps = check.steps
        if check.scope is SLOScope.EACH and check.passed:
            # Passing 'each' checks record every examined hop.
            check_steps = tuple(sv.step for sv in check.step_values)
        tokens.append(f"slo_{check.key}={'pass' if check.passed else 'fail'}")
        tokens.append(f"slo_{check.key}_steps={','.join(str(n) for n in check_steps)}")
        if check.baseline_ms is not None:
            tokens.append(f"slo_{check.key}_baseline={check.baseline_ms:.1f}")
            tokens.append(f"slo_{check.key}_current={check.current_ms:.1f}")
            if check.relative_pct is not None:
                tokens.append(f"slo_{check.key}_pct={check.relative_pct:.1f}")
    return tokens


def format_compact_line(step: StepMetrics) -> str:
    """Format step metrics as a human-readable single-line summary.

    Unlike :func:`format_metrics_line` (which targets scripting and
    emits ``key=value`` tokens with no units), this format is meant
    for terminal logs and pipelines where a person will read the line:
    timings carry ``ms`` suffixes, response size is rendered with an
    appropriate unit, and the line leads with the HTTP status so
    failures stand out when grepping.

    Args:
        step: Step metrics to format.

    Returns:
        Space-separated, human-readable summary on a single line.

    Examples:
        >>> format_compact_line(step)
        'Step 1: 200 GET https://example.com | dns=1.5ms connect=45.2ms tls=67.8ms ttfb=156.4ms total=234.7ms | 1.2 KB'

    """
    status = step.response.status if step.response.status is not None else "—"
    method = step.request_method or "GET"

    timings = " ".join(
        [
            f"dns={step.timing.dns_ms:.1f}ms",
            f"connect={step.timing.connect_ms:.1f}ms",
            f"tls={step.timing.tls_ms:.1f}ms",
            f"ttfb={step.timing.ttfb_ms:.1f}ms",
            f"total={step.timing.total_ms:.1f}ms",
        ]
    )

    size = format_bytes_human(step.response.bytes)

    return f"Step {step.step_number}: {status} {method} {step.url} | {timings} | {size}"


def format_slo_panel(result: SLOResult) -> Panel:
    """Render SLO budgets and any violations as a Rich panel.

    The legacy path (default ``final`` scope, absolute budgets, no
    baseline) renders exactly as before. Enriched targets additionally
    show the scope, resolved thresholds, baseline comparison figures,
    and the hops each check concerns.

    Args:
        result: Evaluated SLO result.

    Returns:
        A Rich ``Panel`` suitable for printing after the waterfall.

    """
    title = "[bold green]✓ SLO: pass[/bold green]" if result.passed else "[bold red]✗ SLO: fail[/bold red]"
    border = "green" if result.passed else "red"

    body = Text()
    if result.is_legacy:
        body.append("Thresholds: ", style="bold")
        if result.thresholds_ms:
            parts = [f"{key}≤{value:g}ms" for key, value in sorted(result.thresholds_ms.items())]
            body.append(", ".join(parts))
        else:
            body.append("(none)", style="dim")

        if not result.passed:
            body.append("\n")
            body.append("Violations:\n", style="bold red")
            for violation in result.violations:
                body.append(
                    f"  • {violation.key}: "
                    f"{violation.actual_ms:.1f}ms > {violation.threshold_ms:g}ms "
                    f"(+{violation.delta_ms:.1f}ms)\n",
                    style="red",
                )

        return Panel(body, title=title, border_style=border, padding=(0, 1))

    body.append("Scope: ", style="bold")
    body.append(_scope_description(result.scope))

    body.append("Budgets: ", style="bold")
    budget_parts = [_budget_description(check) for check in result.checks]
    body.append(", ".join(budget_parts) if budget_parts else "(none)", style="dim" if not budget_parts else "")
    body.append(" ")

    # Measured values vs baseline are audited on pass as well as fail.
    if any(check.baseline_ms is not None for check in result.checks):
        body.append("Results: ", style="bold")
        body.append(", ".join(_result_description(check) for check in result.checks))

    if not result.passed:
        body.append("\n")
        body.append("Violations:\n", style="bold red")
        for violation, check in zip(result.violations, _checks_by_key(result, result.violations), strict=True):
            body.append(_format_enriched_violation(violation, check), style="red")

    return Panel(body, title=title, border_style=border, padding=(0, 1))


def _scope_description(scope: SLOScope) -> str:
    """Return one-line human description of a non-default scope."""
    if scope is SLOScope.CHAIN:
        return "chain — each phase aggregated across every successful hop (rule: sum).\n"
    return "each — every hop checked against the budgets individually.\n"


def _budget_description(check: SLOCheck) -> str:
    """Render one resolved budget for the panel body."""
    if check.budget_pct is not None:
        sign = "+" if check.budget_pct > 0 else ""
        base = check.step_values[0].baseline_ms if check.scope is SLOScope.FINAL else check.baseline_ms
        return f"{check.key}≤{sign}{check.budget_pct:g}% (baseline {base:.1f}ms → limit {check.threshold_ms:.1f}ms)"
    return f"{check.key}≤{check.threshold_ms:g}ms"


def _result_description(check: SLOCheck) -> str:
    """Render the measured outcome of one check vs its baseline."""
    if check.relative_pct is not None:
        sign = "+" if check.relative_pct > 0 else ""
        pct = f"{sign}{check.relative_pct:.1f}%"
    else:
        pct = "undefined%"  # Baseline value was zero.
    return f"{check.key}: {check.current_ms:.1f}ms vs baseline {check.baseline_ms:.1f}ms ({pct})"


def _checks_by_key(
    result: SLOResult,
    violations: Sequence[SLOViolation],
) -> list[SLOCheck]:
    """Look up the check record paired with each violation (keyed)."""
    by_key = {check.key: check for check in result.checks}
    return [by_key[v.key] for v in violations]


def _format_enriched_violation(violation: SLOViolation, check: SLOCheck) -> str:
    """Render one violation line including scope/baseline/step details."""
    line = (
        f"  • {violation.key}: {violation.actual_ms:.1f}ms > {violation.threshold_ms:.1f}ms "
        f"(+{violation.delta_ms:.1f}ms over budget)"
    )
    if violation.baseline_ms is not None and violation.relative_pct is not None:
        sign = "+" if violation.relative_pct > 0 else ""
        line += f" | baseline {violation.baseline_ms:.1f}ms, {sign}{violation.relative_pct:.1f}%"
    line += "\n"

    if check.scope is SLOScope.CHAIN:
        hops = " ".join(f"{sv.step}={sv.current_ms:.1f}ms" for sv in check.step_values)
        line += f"      hops: {hops}\n"
    elif check.scope is SLOScope.EACH:
        bad = set(violation.steps)
        hops = " ".join(f"{sv.step}={sv.current_ms:.1f}ms" for sv in check.step_values if sv.step in bad)
        line += f"      violating hops: {hops}\n"
    elif violation.steps:
        line += f"      step {violation.steps[0]}\n"

    return line
