"""Chart utilities for the Streamlit app."""

import altair as alt
import pandas as pd
from typing import Any, Mapping, Sequence


_LANG_NAME_MAP: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "unknown": "Unknown",
}


def daily_volume_chart(daily_metrics: pd.DataFrame) -> alt.Chart:
    """Create daily volume chart (traces, users, threads)."""
    vol_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["traces", "unique_users", "unique_threads"],
        var_name="metric",
        value_name="value",
    )
    vol_long["metric"] = vol_long["metric"].replace({
        "traces": "Traces",
        "unique_users": "Unique users",
        "unique_threads": "Unique threads",
    })
    return (
        alt.Chart(vol_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Count"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Count", format=","),
            ],
        )
        .properties(title="Daily volume (traces, users, threads)")
    )


def daily_outcome_chart(
    daily_metrics: pd.DataFrame,
    *,
    outcome_order: Sequence[str] | None = None,
    outcome_colors: Mapping[str, str] | None = None,
) -> alt.Chart:
    """Create daily outcome rates stacked area chart."""
    outcome_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["error_rate", "empty_rate", "defer_rate", "soft_error_rate", "success_rate"],
        var_name="metric",
        value_name="value",
    )
    outcome_long["metric"] = outcome_long["metric"].replace({
        "error_rate": "Error",
        "empty_rate": "Error (Empty)",
        "defer_rate": "Defer",
        "soft_error_rate": "Soft error",
        "success_rate": "Success",
    })

    order = list(outcome_order) if outcome_order is not None else [
        "Success",
        "Soft error",
        "Defer",
        "Error (Empty)",
        "Error",
    ]

    default_colors: dict[str, str] = {
        "Success": "#0068C9",
        "Soft error": "#83C9FF",
        "Defer": "#D5DAE5",
        "Error (Empty)": "#FFABAB",
        "Error": "#FF2B2B",
    }
    color_map = {**default_colors, **(dict(outcome_colors) if outcome_colors is not None else {})}
    color_scale = alt.Scale(domain=order, range=[color_map.get(k, "#999999") for k in order])
    return (
        alt.Chart(outcome_long)
        .mark_area()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Rate", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", title="Outcome", sort=order, scale=color_scale),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Outcome"),
                alt.Tooltip("value:Q", title="Rate", format=".1%"),
            ],
        )
        .properties(title="Daily outcome rates")
    )


def daily_cost_chart(daily_metrics: pd.DataFrame) -> alt.Chart:
    """Create daily cost chart."""
    cost_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["mean_cost", "p95_cost"],
        var_name="metric",
        value_name="value",
    )
    cost_long["metric"] = cost_long["metric"].replace({"mean_cost": "Mean cost", "p95_cost": "p95 cost"})
    return (
        alt.Chart(cost_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="USD"),
            color=alt.Color("metric:N", title="Cost"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Cost type"),
                alt.Tooltip("value:Q", title="USD", format="$.4f"),
            ],
        )
        .properties(title="Daily cost")
    )


def daily_latency_chart(daily_metrics: pd.DataFrame) -> alt.Chart:
    """Create daily latency chart."""
    lat_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["mean_latency", "p95_latency"],
        var_name="metric",
        value_name="value",
    )
    lat_long["metric"] = lat_long["metric"].replace({"mean_latency": "Mean latency", "p95_latency": "p95 latency"})
    return (
        alt.Chart(lat_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Seconds"),
            color=alt.Color("metric:N", title="Latency"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Latency type"),
                alt.Tooltip("value:Q", title="Seconds", format=".2f"),
            ],
        )
        .properties(title="Daily latency")
    )
    
def outcome_pie_chart(df: pd.DataFrame) -> alt.Chart:
    """Create outcome breakdown pie chart."""
    outcome_counts = df["outcome"].value_counts().reset_index()
    outcome_counts.columns = ["outcome", "count"]

    # Align labels + colors with daily_outcome_chart
    outcome_counts["outcome"] = outcome_counts["outcome"].replace(
        {
            "ANSWER": "Success",
            "DEFER": "Defer",
            "SOFT_ERROR": "Soft error",
            "EMPTY": "Error (Empty)",
            "ERROR": "Error",
        }
    )

    order = ["Error", "Error (Empty)", "Defer", "Soft error", "Success"]
    default_colors: dict[str, str] = {
        "Success": "#0068C9",
        "Soft error": "#83C9FF",
        "Defer": "#D5DAE5",
        "Error (Empty)": "#FFABAB",
        "Error": "#FF2B2B",
    }
    color_scale = alt.Scale(domain=order, range=[default_colors.get(k, "#999999") for k in order])

    total = max(1, int(outcome_counts["count"].sum()))
    outcome_counts["percent"] = (outcome_counts["count"] / total * 100).round(1)
    return (
        alt.Chart(outcome_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("count:Q", title="Count"),
            color=alt.Color("outcome:N", title="Outcome", sort=order, scale=color_scale),
            tooltip=[
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Count", format=","),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title="Outcome breakdown", height=250)
    )


def language_bar_chart(df: pd.DataFrame, top_n: int = 15) -> alt.Chart | None:
    """Create top languages bar chart."""
    if "lang_query" not in df.columns:
        return None
    lang_counts = (
        df["lang_query"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .value_counts()
        .head(top_n)
    )
    if not len(lang_counts):
        return None
    lang_df = lang_counts.rename_axis("language").reset_index(name="count")
    lang_df["language_name"] = (
        lang_df["language"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda c: _LANG_NAME_MAP.get(c, c or "Unknown"))
    )
    lang_df["percent"] = (lang_df["count"] / lang_df["count"].sum() * 100).round(1)
    return (
        alt.Chart(lang_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("language:N", sort="-x", title="Language"),
            tooltip=[
                alt.Tooltip("language_name:N", title="Language"),
                alt.Tooltip("language:N", title="Code"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title="Top prompt languages (langid)", height=dynamic_bar_height(len(lang_df), bar_px=24, min_px=180))
    )


def latency_histogram(latency_series: pd.Series) -> alt.Chart | None:
    """Create latency distribution histogram with proper bin tooltips."""
    if not len(latency_series):
        return None
    lat_df = pd.DataFrame({"latency_seconds": latency_series})
    return (
        alt.Chart(lat_df)
        .transform_bin(
            as_=["bin_start", "bin_end"],
            field="latency_seconds",
            bin=alt.Bin(maxbins=30),
        )
        .transform_aggregate(
            count="count()",
            groupby=["bin_start", "bin_end"],
        )
        .transform_calculate(
            bin_range="format(datum.bin_start, '.1f') + 's – ' + format(datum.bin_end, '.1f') + 's'"
        )
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title="Latency (s)", bin="binned"),
            x2=alt.X2("bin_end:Q"),
            y=alt.Y("count:Q", title="Traces"),
            tooltip=[
                alt.Tooltip("bin_range:N", title="Latency range"),
                alt.Tooltip("count:Q", title="Traces", format=","),
            ],
        )
        .properties(title="Latency distribution", height=220)
    )


def cost_histogram(cost_series: pd.Series) -> alt.Chart | None:
    """Create cost distribution histogram with proper bin tooltips."""
    if not len(cost_series):
        return None
    cost_df = pd.DataFrame({"total_cost": cost_series})
    return (
        alt.Chart(cost_df)
        .transform_bin(
            as_=["bin_start", "bin_end"],
            field="total_cost",
            bin=alt.Bin(maxbins=30),
        )
        .transform_aggregate(
            count="count()",
            groupby=["bin_start", "bin_end"],
        )
        .transform_calculate(
            bin_range="'$' + format(datum.bin_start, '.4f') + ' – $' + format(datum.bin_end, '.4f')"
        )
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title="Total cost (USD)", bin="binned"),
            x2=alt.X2("bin_end:Q"),
            y=alt.Y("count:Q", title="Traces"),
            tooltip=[
                alt.Tooltip("bin_range:N", title="Cost range"),
                alt.Tooltip("count:Q", title="Traces", format=","),
            ],
        )
        .properties(title="Cost distribution", height=220)
    )


def category_pie_chart(
    series: pd.Series,
    label: str,
    title: str,
    top_n: int = 10,
    explode_csv: bool = False,
) -> alt.Chart | None:
    """Create a pie chart for a categorical series."""
    if explode_csv:
        counts = (
            series.fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .astype(str)
            .str.strip()
            .replace({"": None})
            .dropna()
            .value_counts()
            .head(top_n)
        )
    else:
        counts = (
            series.fillna("")
            .astype(str)
            .str.strip()
            .replace({"": None})
            .dropna()
            .value_counts()
            .head(top_n)
        )
    if not len(counts):
        return None
    df = counts.rename_axis(label).reset_index(name="count")
    df["percent"] = (df["count"] / df["count"].sum() * 100).round(1)
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(f"{label}:N", title=title),
            tooltip=[
                alt.Tooltip(f"{label}:N", title=title),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title=title, height=250)
    )


def success_rate_bar_chart(
    metric_df: pd.DataFrame,
    group_col: str,
    title: str,
) -> alt.Chart | None:
    """Create success rate bar chart from a metric table."""
    if not len(metric_df):
        return None
    return (
        alt.Chart(metric_df)
        .mark_bar()
        .encode(
            x=alt.X("success_rate:Q", title="Success rate", axis=alt.Axis(format="%")),
            y=alt.Y(f"{group_col}:N", sort="-x", title=title),
            tooltip=[
                alt.Tooltip(f"{group_col}:N", title=title),
                alt.Tooltip("traces:Q", title="Traces", format=","),
                alt.Tooltip("success_rate:Q", title="Success rate", format=".1%"),
                alt.Tooltip("defer_rate:Q", title="Defer rate", format=".1%"),
                alt.Tooltip("soft_error_rate:Q", title="Soft error rate", format=".1%"),
                alt.Tooltip("error_rate:Q", title="Error rate", format=".1%"),
                alt.Tooltip("empty_rate:Q", title="Empty rate", format=".1%"),
            ],
        )
        .properties(title=f"Success rate by {title.lower()}")
    )


# ---------------------------------------------------------------------------
# Agentic / Tool Analysis Charts
# ---------------------------------------------------------------------------

def tool_success_rate_chart(tool_stats_df: pd.DataFrame) -> alt.Chart | None:
    """Create stacked bar chart of tool call outcomes by tool name.

    Expects df with columns: tool_name, success, ambiguity, semantic_error, error
    """
    if not len(tool_stats_df):
        return None

    # Melt to long form for stacking
    long_df = tool_stats_df.melt(
        id_vars=["tool_name", "total"],
        value_vars=["success", "ambiguity", "semantic_error", "error"],
        var_name="outcome",
        value_name="count",
    )
    long_df["outcome"] = long_df["outcome"].replace({
        "success": "Success",
        "ambiguity": "Ambiguity",
        "semantic_error": "Semantic Error",
        "error": "Error",
    })

    # Calculate percentage for tooltip
    long_df["pct"] = long_df["count"] / long_df["total"].clip(lower=1)

    color_scale = alt.Scale(
        domain=["Success", "Ambiguity", "Semantic Error", "Error"],
        range=["#4CAF50", "#FFC107", "#FF9800", "#F44336"],
    )

    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Tool calls", stack="zero"),
            y=alt.Y("tool_name:N", sort="-x", title="Tool"),
            color=alt.Color("outcome:N", title="Outcome", scale=color_scale),
            tooltip=[
                alt.Tooltip("tool_name:N", title="Tool"),
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Count", format=","),
                alt.Tooltip("pct:Q", title="% of tool calls", format=".1%"),
                alt.Tooltip("total:Q", title="Total calls", format=","),
            ],
        )
        .properties(title="Tool call outcomes by tool", height=dynamic_bar_height(len(tool_stats_df), bar_px=38, min_px=260))
    )


def tool_calls_vs_latency_chart(df: pd.DataFrame) -> alt.Chart | None:
    """Create scatter plot of tool call count vs latency per trace.

    Expects df with columns: tool_call_count, latency_seconds, outcome (optional)
    """
    if not len(df) or "tool_call_count" not in df.columns or "latency_seconds" not in df.columns:
        return None

    plot_df = df[["tool_call_count", "latency_seconds"]].copy()
    plot_df["tool_call_count"] = pd.to_numeric(plot_df["tool_call_count"], errors="coerce")
    plot_df["latency_seconds"] = pd.to_numeric(plot_df["latency_seconds"], errors="coerce")
    plot_df = plot_df.dropna(subset=["tool_call_count", "latency_seconds"]).copy()
    if "outcome" in df.columns:
        plot_df["outcome"] = df.loc[plot_df.index, "outcome"]
    else:
        plot_df["outcome"] = "Unknown"

    if not len(plot_df):
        return None

    base = alt.Chart(plot_df)

    points = base.mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("tool_call_count:Q", title="Tool calls per trace"),
        y=alt.Y("latency_seconds:Q", title="Latency (s)"),
        color=alt.Color("outcome:N", title="Outcome"),
        tooltip=[
            alt.Tooltip("tool_call_count:Q", title="Tool calls"),
            alt.Tooltip("latency_seconds:Q", title="Latency (s)", format=".2f"),
            alt.Tooltip("outcome:N", title="Outcome"),
        ],
    )

    chart: alt.Chart = points

    # Add trend line (only if regression is well-defined)
    try:
        if len(plot_df) >= 2 and plot_df["tool_call_count"].nunique() >= 2:
            trend = (
                base.transform_regression("tool_call_count", "latency_seconds")
                .mark_line(color="gray", strokeDash=[4, 4], opacity=0.7)
            )
            chart = points + trend
    except Exception:
        chart = points

    return chart.properties(
        title="Tool calls vs latency",
        height=280,
    )


def reasoning_tokens_histogram(reasoning_ratios: pd.Series) -> alt.Chart | None:
    """Create histogram of reasoning token ratio (reasoning / output tokens).

    Expects a series of ratios between 0 and 1.
    """
    ratios = reasoning_ratios.dropna()
    ratios = ratios[ratios >= 0]
    if not len(ratios):
        return None

    ratio_df = pd.DataFrame({"reasoning_ratio": ratios})

    return (
        alt.Chart(ratio_df)
        .transform_bin(
            as_=["bin_start", "bin_end"],
            field="reasoning_ratio",
            bin=alt.Bin(maxbins=20, extent=[0, 1]),
        )
        .transform_aggregate(
            count="count()",
            groupby=["bin_start", "bin_end"],
        )
        .transform_calculate(
            bin_range="format(datum.bin_start * 100, '.0f') + '% – ' + format(datum.bin_end * 100, '.0f') + '%'"
        )
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title="Reasoning tokens (% of output)", bin="binned", axis=alt.Axis(format="%")),
            x2=alt.X2("bin_end:Q"),
            y=alt.Y("count:Q", title="Traces"),
            tooltip=[
                alt.Tooltip("bin_range:N", title="Reasoning ratio"),
                alt.Tooltip("count:Q", title="Traces", format=","),
            ],
        )
        .properties(title="Reasoning tokens share distribution", height=220)
    )


def tool_flow_sankey_data(traces: list, extract_flow_fn) -> pd.DataFrame:
    """Build edge list for tool flow visualization.

    Returns df with columns: source, target, count, status
    Where source/target are "START", tool names, or "END".
    """
    edges: dict[tuple[str, str, str], int] = {}

    for t in traces:
        flow = extract_flow_fn(t)
        if not flow:
            continue

        # START -> first tool
        first_tool, first_status = flow[0]
        key = ("START", first_tool, first_status)
        edges[key] = edges.get(key, 0) + 1

        # tool -> tool transitions
        for i in range(len(flow) - 1):
            curr_tool, _ = flow[i]
            next_tool, next_status = flow[i + 1]
            key = (curr_tool, next_tool, next_status)
            edges[key] = edges.get(key, 0) + 1

        # last tool -> END
        last_tool, last_status = flow[-1]
        key = (last_tool, "END", last_status)
        edges[key] = edges.get(key, 0) + 1

    if not edges:
        return pd.DataFrame()

    rows = [
        {"source": s, "target": t, "status": st, "count": c}
        for (s, t, st), c in edges.items()
    ]
    return pd.DataFrame(rows)


def dynamic_bar_height(n_entries: int, bar_px: int = 28, min_px: int = 180, max_px: int = 800) -> int:
    """Return a chart height that scales with the number of bar entries."""
    return max(min_px, min(max_px, n_entries * bar_px))


# ---------------------------------------------------------------------------
# Prompt utilisation charts
# ---------------------------------------------------------------------------


def prompt_utilisation_histogram(user_day_counts: pd.DataFrame) -> alt.Chart:
    """Histogram of prompts-per-user-per-day distribution."""
    maxbins = 30
    bar_size = max(6, min(28, int(600 / max(1, maxbins))))
    return (
        alt.Chart(user_day_counts)
        .mark_bar(size=bar_size)
        .encode(
            x=alt.X(
                "prompts:Q",
                bin=alt.Bin(maxbins=maxbins),
                title="Prompts per user per day",
            ),
            y=alt.Y("count():Q", title="User-days"),
            tooltip=[
                alt.Tooltip("prompts:Q", title="Prompts/user/day", bin=True),
                alt.Tooltip("count():Q", title="User-days"),
            ],
        )
        .properties(height=260)
    )


def prompt_utilisation_daily_chart(
    daily_user_prompt: pd.DataFrame,
    prompt_limit: int = 10,
) -> alt.LayerChart:
    """Daily mean/median/p95 prompts per active user with a limit rule."""
    daily_long = daily_user_prompt.melt(
        id_vars=["date"],
        value_vars=[
            "mean_prompts_per_user",
            "median_prompts_per_user",
            "p95_prompts_per_user",
        ],
        var_name="metric",
        value_name="value",
    )
    daily_long["metric"] = daily_long["metric"].replace(
        {
            "mean_prompts_per_user": "Mean",
            "median_prompts_per_user": "Median",
            "p95_prompts_per_user": "p95",
        }
    )

    line = (
        alt.Chart(daily_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Prompts per user"),
            color=alt.Color("metric:N", title="Metric", sort=["Mean", "Median", "p95"]),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Prompts/user", format=".2f"),
            ],
        )
        .properties(height=260)
    )

    limit_rule = (
        alt.Chart(pd.DataFrame({"prompt_limit": [prompt_limit]}))
        .mark_rule(color="#e5484d", strokeDash=[6, 4])
        .encode(y=alt.Y("prompt_limit:Q"))
    )

    return line + limit_rule


# ---------------------------------------------------------------------------
# User segment charts
# ---------------------------------------------------------------------------


def user_segment_bar_chart(
    daily_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
) -> alt.Chart:
    """Stacked bar chart for a two-category user segment over time."""
    day_total = daily_df[col_a] + daily_df[col_b]
    long = daily_df.assign(day_total=day_total).melt(
        id_vars=["date", "day_total"],
        value_vars=[col_a, col_b],
        var_name="user_type",
        value_name="count",
    )
    long["pct_of_day"] = long["count"] / long["day_total"].clip(lower=1)
    long["user_type"] = long["user_type"].replace({col_a: label_a, col_b: label_b})

    domain = [label_a, label_b]
    scale = alt.Scale(domain=domain, range=[color_a, color_b])

    n_days = int(daily_df["date"].nunique()) if "date" in daily_df.columns else 1
    bar_size = max(3, min(20, int(420 / max(1, n_days))))

    return (
        alt.Chart(long)
        .mark_bar(size=bar_size)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title="Users", stack=True),
            color=alt.Color("user_type:N", title="User type", sort=domain, scale=scale),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("user_type:N", title="Type"),
                alt.Tooltip("count:Q", title="Users", format=","),
                alt.Tooltip("pct_of_day:Q", title="% of day", format=".1%"),
            ],
        )
        .properties(height=220)
    )


def user_segment_pie_chart(
    label_a: str,
    label_b: str,
    count_a: int,
    count_b: int,
    color_a: str,
    color_b: str,
) -> alt.Chart:
    """Donut chart for a two-category user segment total."""
    domain = [label_a, label_b]
    scale = alt.Scale(domain=domain, range=[color_a, color_b])

    pie_df = pd.DataFrame([
        {"user_type": label_a, "count": count_a},
        {"user_type": label_b, "count": count_b},
    ])
    pie_df = pie_df[pie_df["count"] > 0]
    pie_df["percent"] = pie_df["count"] / max(1, int(pie_df["count"].sum())) * 100

    return (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta("count:Q", title="Users"),
            color=alt.Color("user_type:N", title="", sort=domain, scale=scale),
            tooltip=[
                alt.Tooltip("user_type:N", title="Type"),
                alt.Tooltip("count:Q", title="Users", format=","),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(height=220)
    )


# ---------------------------------------------------------------------------
# Starter-prompt charts
# ---------------------------------------------------------------------------


def starter_vs_other_pie(starter_count: int, other_count: int) -> alt.Chart:
    """Donut chart of starter prompts vs other prompts."""
    df = pd.DataFrame([
        {"label": "Starter", "count": starter_count},
        {"label": "Other", "count": other_count},
    ])
    df["percent"] = (df["count"] / max(1, df["count"].sum()) * 100).round(1)
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta("count:Q", title="Count"),
            color=alt.Color("label:N", title=""),
            tooltip=[
                alt.Tooltip("label:N", title="Type"),
                alt.Tooltip("count:Q", title="Count", format=","),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(height=260)
    )


def starter_breakdown_pie(breakdown_df: pd.DataFrame) -> alt.Chart:
    """Donut chart breaking down individual starter prompt usage.

    Expects df with columns: label, count.
    """
    df = breakdown_df.copy()
    df["percent"] = (df["count"] / max(1, df["count"].sum()) * 100).round(1)
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta("count:Q", title="Count"),
            color=alt.Color("label:N", title=""),
            tooltip=[
                alt.Tooltip("label:N", title="Starter prompt"),
                alt.Tooltip("count:Q", title="Count", format=","),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(height=260)
    )


# ---------------------------------------------------------------------------
# Prompt-length histograms
# ---------------------------------------------------------------------------


def prompt_length_histogram(
    series: pd.Series,
    field: str,
    title: str,
) -> alt.Chart:
    """Generic histogram for prompt length (chars or words)."""
    df = pd.DataFrame({field: series})
    return (
        alt.Chart(df)
        .transform_bin(
            as_=["bin_start", "bin_end"],
            field=field,
            bin=alt.Bin(maxbins=60),
        )
        .transform_calculate(bin_width="datum.bin_end - datum.bin_start")
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title=title, bin=alt.Bin(binned=True)),
            x2=alt.X2("bin_end:Q"),
            y=alt.Y("count()", title="Count"),
            tooltip=[
                alt.Tooltip("bin_start:Q", title="Bin start", format=","),
                alt.Tooltip("bin_end:Q", title="Bin end", format=","),
                alt.Tooltip("bin_width:Q", title="Bin width", format=","),
                alt.Tooltip("count()", title="Count", format=","),
            ],
        )
        .properties(height=180)
    )


# ---------------------------------------------------------------------------
# AOI charts
# ---------------------------------------------------------------------------


def aoi_type_pie(aoi_type_df: pd.DataFrame, domain: list[str] | None = None) -> alt.Chart:
    """Donut chart for AOI type distribution.

    Expects df with columns: aoi_type, count, percent.
    """
    scale = alt.Scale(domain=domain, scheme="tableau10") if domain else alt.Scale(scheme="tableau10")
    return (
        alt.Chart(aoi_type_df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("aoi_type:N", title="AOI type", scale=scale),
            tooltip=[
                alt.Tooltip("aoi_type:N", title="AOI type"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title="AOI type", height=250)
    )


def aoi_name_bar(
    aoi_name_df: pd.DataFrame,
    aoi_type_domain: list[str] | None = None,
) -> alt.Chart:
    """Horizontal bar chart of top AOI names, optionally colored by AOI type.

    Expects df with columns: aoi_name, count, and optionally aoi_type.
    """
    n = len(aoi_name_df)
    height = dynamic_bar_height(n, bar_px=22, min_px=200, max_px=800)

    enc_color: alt.Color | None = None
    if "aoi_type" in aoi_name_df.columns and aoi_type_domain:
        enc_color = alt.Color(
            "aoi_type:N",
            title="AOI type",
            scale=alt.Scale(domain=aoi_type_domain, scheme="tableau10"),
        )
    elif "aoi_type" in aoi_name_df.columns:
        enc_color = alt.Color("aoi_type:N", title="AOI type")

    tooltip_fields = [
        alt.Tooltip("aoi_name:N", title="AOI"),
        alt.Tooltip("count:Q", title="Count", format=","),
    ]
    if "aoi_type" in aoi_name_df.columns:
        tooltip_fields.insert(1, alt.Tooltip("aoi_type:N", title="AOI type"))

    encoding: dict[str, Any] = {
        "x": alt.X("count:Q", title="Count"),
        "y": alt.Y("aoi_name:N", sort="-x", title="AOI"),
        "tooltip": tooltip_fields,
    }
    if enc_color is not None:
        encoding["color"] = enc_color

    return (
        alt.Chart(aoi_name_df)
        .mark_bar()
        .encode(**encoding)
        .properties(title="AOI selection counts", height=height)
    )


# ---------------------------------------------------------------------------
# Error / flow pie charts
# ---------------------------------------------------------------------------


def simple_pie_chart(
    df: pd.DataFrame,
    label_col: str = "label",
    count_col: str = "count",
    *,
    height: int = 220,
) -> alt.Chart:
    """Generic donut chart from a label + count DataFrame.

    Adds a percent column automatically.
    """
    df = df.copy()
    df = df[df[count_col] > 0]
    total = max(1, int(df[count_col].sum()))
    df["percent"] = (df[count_col] / total * 100).round(1)

    return (
        alt.Chart(df)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta(f"{count_col}:Q", title="Count"),
            color=alt.Color(f"{label_col}:N", title=""),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title=""),
                alt.Tooltip(f"{count_col}:Q", title="Count", format=","),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(height=height)
    )


def tool_flow_arc_chart(flow_df: pd.DataFrame) -> alt.Chart | None:
    """Create arc diagram showing tool call flows with status coloring.

    This is an alternative to Sankey that works in Altair.
    """
    if not len(flow_df):
        return None

    # Get unique nodes and assign positions
    nodes = sorted(set(flow_df["source"].tolist() + flow_df["target"].tolist()))
    node_order = {"START": 0, "END": 999}
    for i, n in enumerate(nodes):
        if n not in node_order:
            node_order[n] = i + 1
    nodes = sorted(nodes, key=lambda x: node_order.get(x, 100))

    node_df = pd.DataFrame({
        "node": nodes,
        "x": list(range(len(nodes))),
    })

    # Add source/target x positions to flow_df
    flow_df = flow_df.copy()
    flow_df["source_x"] = flow_df["source"].map(lambda n: node_order.get(n, 100))
    flow_df["target_x"] = flow_df["target"].map(lambda n: node_order.get(n, 100))

    color_scale = alt.Scale(
        domain=["success", "ambiguity", "semantic_error", "error"],
        range=["#4CAF50", "#FFC107", "#FF9800", "#F44336"],
    )

    status_order = ["success", "ambiguity", "semantic_error", "error"]

    # Node labels
    node_chart = (
        alt.Chart(node_df)
        .mark_text(fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("x:Q", axis=None),
            text=alt.Text("node:N"),
        )
    )

    # Arcs as rule marks between nodes
    arc_chart = (
        alt.Chart(flow_df)
        .mark_rule(opacity=0.6)
        .encode(
            x=alt.X("source_x:Q", axis=None),
            x2=alt.X2("target_x:Q"),
            y=alt.Y("status:N", title=None, sort=status_order),
            y2=alt.Y2("status:N"),
            color=alt.Color("status:N", title="Status", scale=color_scale),
            strokeWidth=alt.StrokeWidth("count:Q", title="Count", scale=alt.Scale(range=[1, 8])),
            tooltip=[
                alt.Tooltip("source:N", title="From"),
                alt.Tooltip("target:N", title="To"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
    )

    return (
        alt.layer(arc_chart, node_chart)
        .properties(title="Tool call flow", height=420, width="container")
        .configure_view(strokeWidth=0)
    )
