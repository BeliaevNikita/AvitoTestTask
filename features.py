"""Leakage-safe cookie-level feature engineering for the bot challenge."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "cookie_id",
    "event_ts",
    "eid",
    "event_name",
    "platform",
    "user_agent",
    "item_id",
    "item_category",
    "item_location",
    "seller_type",
    "search_query",
    "search_page",
    "pointer_x",
    "pointer_y",
]


def events_in_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Select only window_start_ts <= event_ts < window_end_ts."""
    bounds = meta[["cookie_id", "window_start_ts", "window_end_ts"]]
    if bounds["cookie_id"].duplicated().any():
        raise ValueError("meta.cookie_id must be unique")
    merged = events.merge(bounds, on="cookie_id", how="inner", validate="many_to_one")
    mask = merged["event_ts"].ge(merged["window_start_ts"]) & merged["event_ts"].lt(
        merged["window_end_ts"]
    )
    return merged.loc[mask, EVENT_COLUMNS].copy()


def make_feature_vocab(events: pd.DataFrame) -> dict[str, list[str]]:
    """Create label-free vocabularies from allowed training-window events."""
    platform = events["platform"].str.lower().replace({"iphone": "ios"})
    return {
        "event_name": sorted(events["event_name"].dropna().astype(str).unique()),
        "platform_norm": sorted(platform.dropna().astype(str).unique()),
        "item_category": sorted(events["item_category"].dropna().astype(str).unique()),
        "seller_type": sorted(events["seller_type"].dropna().astype(str).unique()),
    }


def _safe_name(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(value)).strip("_").lower()


def _join(features: pd.DataFrame, values: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return features.join(values, how="left")


def _distribution_stats(events: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    nonnull = events.loc[events[column].notna(), ["cookie_id", column]]
    if nonnull.empty:
        return pd.DataFrame()

    counts = nonnull.groupby(["cookie_id", column], observed=True).size().rename("count")
    totals = counts.groupby(level=0).sum()
    probabilities = counts / counts.groupby(level=0).transform("sum")
    entropy = (-(probabilities * np.log(probabilities))).groupby(level=0).sum()
    nunique = counts.groupby(level=0).size()
    top_share = counts.groupby(level=0).max() / totals
    normalized_entropy = entropy / np.log(nunique.where(nunique.gt(1)))

    out = pd.DataFrame(index=totals.index)
    out[f"{prefix}_nonnull_count"] = totals
    out[f"{prefix}_nunique"] = nunique
    out[f"{prefix}_nunique_ratio"] = nunique / totals
    out[f"{prefix}_top_share"] = top_share
    out[f"{prefix}_entropy"] = entropy
    out[f"{prefix}_entropy_norm"] = normalized_entropy.fillna(0.0)
    return out


def _count_share_features(
    events: pd.DataFrame,
    column: str,
    values: Iterable[str],
    prefix: str,
) -> pd.DataFrame:
    table = events.groupby(["cookie_id", column], observed=True).size().unstack(fill_value=0)
    total = events.groupby("cookie_id").size()
    out = pd.DataFrame(index=table.index)
    for value in values:
        count = table[value] if value in table.columns else pd.Series(0, index=table.index)
        name = _safe_name(value)
        out[f"{prefix}_{name}_count"] = count
        out[f"{prefix}_{name}_share"] = count / total
    return out


def _gap_features(sequence: pd.DataFrame, prefix: str) -> pd.DataFrame:
    sequence = sequence.sort_values(["cookie_id", "event_ts"], kind="mergesort").copy()
    sequence["gap_s"] = sequence.groupby("cookie_id", sort=False)["event_ts"].diff().dt.total_seconds()
    gap_group = sequence.groupby("cookie_id", sort=False)["gap_s"]
    out = gap_group.agg(["count", "mean", "std", "min", "max", "median"])
    out.columns = [f"{prefix}_{name}" for name in out.columns]

    for quantile in (0.10, 0.25, 0.75, 0.90, 0.95):
        out[f"{prefix}_q{int(quantile * 100):02d}"] = gap_group.quantile(quantile)
    out[f"{prefix}_cv"] = out[f"{prefix}_std"] / out[f"{prefix}_mean"].replace(0, np.nan)
    out[f"{prefix}_iqr_over_median"] = (
        out[f"{prefix}_q75"] - out[f"{prefix}_q25"]
    ) / out[f"{prefix}_median"].replace(0, np.nan)

    valid = sequence.loc[sequence["gap_s"].notna(), ["cookie_id", "gap_s"]].copy()
    if not valid.empty:
        by_gap = valid.groupby(["cookie_id", "gap_s"], observed=True).size()
        interval_count = by_gap.groupby(level=0).sum()
        gap_nunique = by_gap.groupby(level=0).size()
        out[f"{prefix}_mode_share"] = by_gap.groupby(level=0).max() / interval_count
        out[f"{prefix}_nunique_ratio"] = gap_nunique / interval_count
        for seconds in (5, 10, 30, 60):
            share = valid["gap_s"].mod(seconds).eq(0).groupby(valid["cookie_id"]).mean()
            out[f"{prefix}_multiple_{seconds}s_share"] = share
        out[f"{prefix}_zero_share"] = valid["gap_s"].eq(0).groupby(valid["cookie_id"]).mean()
    return out


def _burst_features(sequence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for cookie_id, part in sequence.groupby("cookie_id", sort=False):
        times = part["event_ts"].astype("int64").to_numpy() / 1_000_000.0
        row: dict[str, float | str] = {"cookie_id": cookie_id}
        starts = np.arange(len(times))
        for seconds in (10, 60, 300):
            ends = np.searchsorted(times, times + seconds, side="left")
            row[f"max_events_{seconds}s"] = float((ends - starts).max())
        rows.append(row)
    return pd.DataFrame(rows).set_index("cookie_id")


def build_cookie_features(
    events: pd.DataFrame,
    meta: pd.DataFrame,
    vocab: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    """Build deterministic numeric features using only each cookie's observation window."""
    ev = events_in_window(events, meta)
    features = meta[["cookie_id"]].drop_duplicates().set_index("cookie_id")

    metadata = meta.set_index("cookie_id")
    age_start = (metadata["window_start_ts"] - metadata["cookie_created_at"]).dt.total_seconds() / 86400
    age_end = (metadata["window_end_ts"] - metadata["cookie_created_at"]).dt.total_seconds() / 86400
    features["cookie_age_start_days"] = age_start
    features["cookie_age_end_days"] = age_end
    features["cookie_age_log1p"] = np.log1p(age_start.clip(lower=0))
    features["cookie_created_hour"] = metadata["cookie_created_at"].dt.hour
    features["cookie_created_dow"] = metadata["cookie_created_at"].dt.dayofweek
    features["window_dow"] = metadata["window_start_ts"].dt.dayofweek
    for days in (1, 7, 30, 90):
        features[f"cookie_age_le_{days}d"] = age_start.le(days).astype(int)

    ev["platform_norm"] = ev["platform"].str.lower().replace({"iphone": "ios"})
    ua = ev["user_agent"].fillna("").str.lower()
    ev["ua_headless"] = ua.str.contains("headless", regex=False)
    ev["ua_automation"] = ua.str.contains(
        "bot|python|requests|curl|wget|scrapy|selenium|playwright", regex=True
    )
    ev["ua_mobile"] = ua.str.contains("mobile|android|iphone|ipad", regex=True)
    ev["ua_linux"] = ua.str.contains("linux", regex=False)
    ev["ua_windows"] = ua.str.contains("windows", regex=False)
    ev["ua_macos"] = ua.str.contains("macintosh|mac os", regex=True)

    total = ev.groupby("cookie_id", sort=False).size().rename("total_events")
    features = _join(features, total)
    features = _join(features, ev.groupby("cookie_id")["event_name"].nunique().rename("event_type_nunique"))

    event_features = _count_share_features(
        ev, "event_name", vocab["event_name"], "event"
    )
    platform_features = _count_share_features(
        ev, "platform_norm", vocab["platform_norm"], "platform"
    )
    category_features = _count_share_features(
        ev, "item_category", vocab["item_category"], "category"
    )
    seller_features = _count_share_features(
        ev, "seller_type", vocab["seller_type"], "seller"
    )
    for block in (event_features, platform_features, category_features, seller_features):
        features = _join(features, block)

    distribution_columns = [
        "event_name",
        "platform_norm",
        "user_agent",
        "item_id",
        "item_category",
        "item_location",
        "seller_type",
        "search_query",
        "search_page",
    ]
    for column in distribution_columns:
        features = _join(
            features,
            _distribution_stats(ev, column, f"{_safe_name(column)}_dist"),
        )

    for flag in (
        "ua_headless",
        "ua_automation",
        "ua_mobile",
        "ua_linux",
        "ua_windows",
        "ua_macos",
    ):
        features = _join(features, ev.groupby("cookie_id")[flag].mean().rename(f"{flag}_share"))

    original_columns = [column for column in EVENT_COLUMNS if column in ev.columns]
    ev["is_exact_duplicate"] = ev.duplicated(subset=original_columns, keep=False)
    features = _join(
        features,
        ev.groupby("cookie_id")["is_exact_duplicate"].agg(
            duplicate_row_count="sum", duplicate_row_share="mean"
        ),
    )
    deduplicated = ev.drop_duplicates(subset=original_columns, keep="first")
    features = _join(
        features,
        deduplicated.groupby("cookie_id").size().rename("deduplicated_events"),
    )

    sequence = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort").copy()
    time_group = sequence.groupby("cookie_id", sort=False)["event_ts"]
    first_event = time_group.min()
    last_event = time_group.max()
    features = _join(features, (last_event - first_event).dt.total_seconds().rename("active_duration_s"))
    features = _join(
        features,
        (first_event - metadata["window_start_ts"]).dt.total_seconds().rename("start_to_first_s"),
    )
    features = _join(
        features,
        (metadata["window_end_ts"] - last_event).dt.total_seconds().rename("last_to_end_s"),
    )
    features = _join(features, _gap_features(sequence, "gap"))
    features = _join(features, _gap_features(deduplicated, "dedup_gap"))

    sequence["event_hour"] = sequence["event_ts"].dt.hour
    sequence["event_minute"] = sequence["event_ts"].dt.floor("min")
    features = _join(
        features,
        sequence.groupby("cookie_id")["event_hour"].nunique().rename("active_hours"),
    )
    features = _join(features, _distribution_stats(sequence, "event_hour", "hour"))

    minute_counts = sequence.groupby(["cookie_id", "event_minute"]).size()
    minute_stats = minute_counts.groupby(level=0).agg(["size", "mean", "std", "max"])
    minute_stats.columns = ["active_minutes", "events_per_active_minute_mean", "events_per_active_minute_std", "max_events_per_minute"]
    minute_stats["top_minute_event_share"] = minute_counts.groupby(level=0).max() / total
    minute_stats["burst_minutes_ge_3"] = minute_counts.ge(3).groupby(level=0).sum()
    minute_stats["burst_minutes_ge_5"] = minute_counts.ge(5).groupby(level=0).sum()
    features = _join(features, minute_stats)
    features = _join(features, _burst_features(sequence))

    previous_event = sequence.groupby("cookie_id", sort=False)["event_name"].shift()
    has_previous = previous_event.notna()
    sequence["same_as_previous"] = sequence["event_name"].eq(previous_event) & has_previous
    features = _join(
        features,
        sequence.loc[has_previous].groupby("cookie_id")["same_as_previous"].mean().rename("repeat_adjacent_share"),
    )
    transition_frame = sequence.loc[has_previous, ["cookie_id", "event_name"]].copy()
    transition_frame["transition"] = previous_event.loc[has_previous] + "__to__" + transition_frame["event_name"]
    features = _join(features, _distribution_stats(transition_frame, "transition", "transition"))

    run_start = previous_event.isna() | sequence["event_name"].ne(previous_event)
    sequence["run_id"] = run_start.groupby(sequence["cookie_id"]).cumsum()
    run_sizes = sequence.groupby(["cookie_id", "run_id"]).size()
    run_stats = run_sizes.groupby(level=0).agg(["size", "mean", "max"])
    run_stats.columns = ["same_action_run_count", "same_action_run_mean", "same_action_run_max"]
    features = _join(features, run_stats)

    gap_s = sequence.groupby("cookie_id", sort=False)["event_ts"].diff().dt.total_seconds()
    sequence["session_id"] = gap_s.gt(1800).groupby(sequence["cookie_id"]).cumsum()
    session_sizes = sequence.groupby(["cookie_id", "session_id"]).size()
    session_stats = session_sizes.groupby(level=0).agg(["size", "mean", "max"])
    session_stats.columns = ["session_count", "session_events_mean", "session_events_max"]
    features = _join(features, session_stats)

    numeric_event_columns = ["search_page", "pointer_x", "pointer_y"]
    for column in numeric_event_columns:
        block = ev.groupby("cookie_id")[column].agg(["count", "mean", "std", "min", "max", "median", "nunique"])
        block.columns = [f"{column}_{name}" for name in block.columns]
        block[f"{column}_present_share"] = block[f"{column}_count"] / total
        features = _join(features, block)

    features = features.replace([np.inf, -np.inf], np.nan).reset_index()
    if not features["cookie_id"].equals(meta["cookie_id"].reset_index(drop=True)):
        raise AssertionError("Feature rows no longer match metadata order")
    return features


def build_error_driven_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Experimental pointer/page transition features kept separate from the final set."""
    sequence = events_in_window(events, meta).sort_values(
        ["cookie_id", "event_ts"], kind="mergesort"
    )
    features = meta[["cookie_id"]].drop_duplicates().set_index("cookie_id")

    pointer_events = sequence.loc[
        sequence["pointer_x"].notna() & sequence["pointer_y"].notna(),
        ["cookie_id", "event_ts", "pointer_x", "pointer_y"],
    ].copy()
    if not pointer_events.empty:
        pointer_group = pointer_events.groupby("cookie_id", sort=False)
        pointer_events["pointer_dx"] = pointer_group["pointer_x"].diff()
        pointer_events["pointer_dy"] = pointer_group["pointer_y"].diff()
        pointer_events["pointer_gap_s"] = pointer_group["event_ts"].diff().dt.total_seconds()
        pointer_events["pointer_distance"] = np.hypot(
            pointer_events["pointer_dx"], pointer_events["pointer_dy"]
        )
        pointer_events["pointer_speed"] = pointer_events["pointer_distance"] / pointer_events[
            "pointer_gap_s"
        ].where(pointer_events["pointer_gap_s"].gt(0))
        pointer_motion = pointer_events.groupby("cookie_id").agg(
            pointer_distance_mean=("pointer_distance", "mean"),
            pointer_distance_median=("pointer_distance", "median"),
            pointer_distance_std=("pointer_distance", "std"),
            pointer_distance_max=("pointer_distance", "max"),
            pointer_speed_mean=("pointer_speed", "mean"),
            pointer_speed_max=("pointer_speed", "max"),
            pointer_x_range=("pointer_x", lambda s: s.max() - s.min()),
            pointer_y_range=("pointer_y", lambda s: s.max() - s.min()),
        )
        pointer_motion["pointer_bbox_area"] = (
            pointer_motion["pointer_x_range"] * pointer_motion["pointer_y_range"]
        )
        valid_pointer_move = pointer_events["pointer_distance"].notna()
        pointer_motion["pointer_zero_distance_share"] = pointer_events.loc[
            valid_pointer_move, "pointer_distance"
        ].eq(0).groupby(pointer_events.loc[valid_pointer_move, "cookie_id"]).mean()
        features = _join(features, pointer_motion)

    for column in ("item_id", "item_category"):
        previous = sequence.groupby("cookie_id", sort=False)[column].shift()
        valid_pair = sequence[column].notna() & previous.notna()
        same_share = sequence.loc[valid_pair, column].eq(previous.loc[valid_pair]).groupby(
            sequence.loc[valid_pair, "cookie_id"]
        ).mean()
        features = _join(features, same_share.rename(f"same_{column}_adjacent_share"))

    page_events = sequence.loc[
        sequence["search_page"].notna(), ["cookie_id", "search_page"]
    ].copy()
    if not page_events.empty:
        page_events["page_diff"] = page_events.groupby("cookie_id")["search_page"].diff()
        page_group = page_events.groupby("cookie_id")["page_diff"]
        page_motion = page_group.agg(["mean", "std", "min", "max", "median"])
        page_motion.columns = [f"search_page_diff_{name}" for name in page_motion.columns]
        valid_page_diff = page_events["page_diff"].notna()
        page_motion["search_page_same_adjacent_share"] = page_events.loc[
            valid_page_diff, "page_diff"
        ].eq(0).groupby(page_events.loc[valid_page_diff, "cookie_id"]).mean()
        page_motion["search_page_forward_share"] = page_events.loc[
            valid_page_diff, "page_diff"
        ].gt(0).groupby(page_events.loc[valid_page_diff, "cookie_id"]).mean()
        features = _join(features, page_motion)

    return features.replace([np.inf, -np.inf], np.nan).reset_index()
