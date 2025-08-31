#!/usr/bin/env python3
"""
Plot timestamped GPS waypoints from a CSV onto a world map image, and optionally
overlay one or more plan paths (lat/lon waypoints connected by lines).

Required telemetry CSV columns (case-insensitive):
    date, time, latitude, longitude, status

Plan CSVs (each row is a separate plan):
    'latlon_plan' column with a string like "[[lat1, lon1], [lat2, lon2], ...]"
    Optional label column among: name, plan, id, label, title

Args:
  fp            (positional): Filepath to the .csv containing waypoints
  --row-start   (optional)  : 1-based inclusive start row in the waypoint CSV
  --row-end     (optional)  : 1-based inclusive end row in the waypoint CSV
  --img-name    (optional)  : Output image filename; defaults to "<csv_basename>_waypoints.png"
  --plan-csv    (optional)  : One or more CSVs with plans (latlon_plan). Each plan is drawn as a connected line.

Examples:
  # Plot all rows
  python vis_comms.py /path/to/points.csv

  # Plot a row slice (1-based, inclusive)
  python vis_comms.py /path/to/points.csv --row-start 200 --row-end 450

  # Plot from row 100 to the end
  python vis_comms.py /path/to/points.csv --row-start 100 --plan-csv plans.csv
"""

import argparse
import ast
import math
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ---- Status color mapping (edit as needed) ----
STATUS_COLORS = {
    "V": "green",   # valid
    "L": "yellow",  # likely
    "G": "orange",  # guess
    "M": "red",     # missed
}
# ------------------------------------------------


def _load_csv(fp: str) -> pd.DataFrame:
    """Load the telemetry CSV with timestamped points. Keeps original row numbers (1-based) in '_orig_row'."""
    df = pd.read_csv(fp)

    # Preserve original row numbers before any cleaning (1-based, like spreadsheets)
    df["_orig_row"] = df.index + 1

    # Case-insensitive rename to ensure required columns exist
    req = {"date", "time", "latitude", "longitude", "status"}
    lower_map = {c.lower(): c for c in df.columns}
    if not req.issubset(lower_map.keys()):
        raise ValueError(f"CSV must contain columns {sorted(req)}; found {list(df.columns)}")
    df = df.rename(columns={lower_map[k]: k for k in req})

    # Combine date + time to a single timestamp (format-agnostic; you can replace with explicit formats if needed)
    # df["timestamp"] = pd.to_datetime(
    #     df["date"].astype(str) + " " + df["time"].astype(str),
    #     errors="coerce"
    # )

    # Drop unusable rows
    df = df.dropna(subset=["latitude", "longitude"]) #"timestamp", 

    return df


def _cartopy_available() -> bool:
    try:
        import cartopy.crs as ccrs  # noqa: F401
        return True
    except Exception:
        return False


def _coerce_pair(pair):
    """Validate a [lat, lon] pair and return (lat, lon) as floats or None if invalid."""
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    lat, lon = pair
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None
    if math.isnan(lat) or math.isnan(lon):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (lat, lon)


def _load_plan_csv(fp: str) -> List[Tuple[pd.DataFrame, str]]:
    """
    Load a plan CSV where EACH ROW represents a separate plan.

    Expected (case-insensitive) columns:
      - 'latlon_plan': a string like "[[lat1, lon1], [lat2, lon2], ...]"
      - optional label column among: 'name', 'plan', 'id', 'label', 'title'
        (if none found, the label defaults to "<basename>_row_<idx>")

    Returns:
      A list of (df, label) tuples, where each df has columns ['lat', 'lon'] for that plan.
    """
    df = pd.read_csv(fp)
    if df.empty:
        return []

    # Case-insensitive column access
    lower_map = {c.lower(): c for c in df.columns}

    # Find the latlon_plan column (required)
    plan_key = None
    for cand in ("latlon_plan", "plan_latlon", "latlon", "lat_lon_plan"):
        if cand in lower_map:
            plan_key = lower_map[cand]
            break
    if not plan_key:
        raise ValueError(
            f"Plan CSV '{fp}' must contain a 'latlon_plan' column (case-insensitive). "
            f"Found columns: {list(df.columns)}"
        )

    # Candidate label columns (optional; first present wins)
    label_key = None
    for cand in ("name", "plan", "id", "label", "title"):
        if cand in lower_map:
            label_key = lower_map[cand]
            break

    out: List[Tuple[pd.DataFrame, str]] = []
    base_label_prefix = os.path.splitext(os.path.basename(fp))[0]

    for idx, row in df.iterrows():
        # Determine label
        if label_key and pd.notna(row[label_key]):
            label = str(row[label_key])
        else:
            label = f"{base_label_prefix}_row_{idx}"

        raw = row.get(plan_key)
        if pd.isna(raw):
            continue

        # Parse the latlon_plan string safely
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            continue

        if not isinstance(parsed, (list, tuple)) or len(parsed) == 0:
            continue

        # Validate and coerce to (lat, lon)
        lat_list, lon_list = [], []
        for p in parsed:
            pair = _coerce_pair(p)
            if pair is not None:
                lat, lon = pair
                lat_list.append(lat)
                lon_list.append(lon)

        if len(lat_list) < 2:
            continue

        plan_df = pd.DataFrame({"lat": lat_list, "lon": lon_list})
        out.append((plan_df, label))

    return out


def _collect_plan_dfs(plan_paths: Optional[List[str]]) -> List[Tuple[pd.DataFrame, str]]:
    plans: List[Tuple[pd.DataFrame, str]] = []
    if not plan_paths:
        return plans
    for p in plan_paths:
        if not p:
            continue
        try:
            plans.extend(_load_plan_csv(p))  # extend because one file may contain multiple plans
        except Exception as e:
            print(f"Warning: could not load plan CSV '{p}': {e}")
    return plans


def _compute_bounds(df_points: pd.DataFrame,
                    plan_dfs: List[Tuple[pd.DataFrame, str]]) -> Tuple[float, float, float, float]:
    """Compute plotting bounds that include both the SELECTED points and any plan paths, with padding."""
    min_lon = df_points["longitude"].min()
    max_lon = df_points["longitude"].max()
    min_lat = df_points["latitude"].min()
    max_lat = df_points["latitude"].max()

    for (pdf, _) in plan_dfs:
        if not pdf.empty:
            min_lon = min(min_lon, pdf["lon"].min())
            max_lon = max(max_lon, pdf["lon"].max())
            min_lat = min(min_lat, pdf["lat"].min())
            max_lat = max(max_lat, pdf["lat"].max())

    pad_lon = max(0.0001, (max_lon - min_lon) * 0.1)
    pad_lat = max(0.0001, (max_lat - min_lat) * 0.1)

    x_min = min_lon - pad_lon
    x_max = max_lon + pad_lon
    y_min = min_lat - pad_lat
    y_max = max_lat + pad_lat
    return x_min, x_max, y_min, y_max


def plot_waypoints(fp: str,
                   row_start: Optional[int],
                   row_end: Optional[int],
                   img_name: Optional[str],
                   plan_paths: Optional[List[str]] = None) -> str:
    df_full = _load_csv(fp)

    # Apply 1-based, inclusive row filtering based on the original CSV row numbers
    df = df_full.copy()
    if row_start is not None and row_start < 1:
        raise ValueError("--row-start must be >= 1 (1-based indexing).")
    if row_end is not None and row_end < 1:
        raise ValueError("--row-end must be >= 1 (1-based indexing).")
    if row_start is not None and row_end is not None and row_end < row_start:
        raise ValueError("--row-end cannot be less than --row-start.")

    if row_start is not None:
        df = df[df["_orig_row"] >= row_start]
    if row_end is not None:
        df = df[df["_orig_row"] <= row_end]

    if df.empty:
        raise ValueError("No points to plot (dataframe is empty after row filtering).")

    # Output path
    if not img_name:
        base, _ = os.path.splitext(fp)
        img_name = base + "_waypoints.png"

    # Load plan CSVs (if any)
    plan_dfs = _collect_plan_dfs(plan_paths)

    # Bounds that include the SELECTED telemetry and plans
    x_min, x_max, y_min, y_max = _compute_bounds(df, plan_dfs)

    use_cartopy = _cartopy_available()
    print("Cartopy available?", use_cartopy)

    # ---- Create figure ----
    plt.figure(figsize=(10, 6))

    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.img_tiles as cimgt

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())

        # Try tile background (requires internet). Fallback to features.
        try:
            tiler = cimgt.QuadtreeTiles()
            ax.add_image(tiler, 17)  # adjust zoom as desired
        except Exception:
            ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="lightblue")
            ax.coastlines(resolution="10m", linewidth=0.8)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5)

        ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
        proj_kwargs = dict(transform=ccrs.PlateCarree())
    else:
        ax = plt.gca()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        proj_kwargs = {}

    # Scatter points by status (color-coded)
    handles = []
    status_dict = {"V": "Valid",
                   "L": "Likely",
                   "G": "Guess",
                   "M": "Missed",
                   }
    # Sort groups by status_dict order
    for status in status_dict.keys():
        group = df[df["status"] == status]
        if not group.empty:
            color = STATUS_COLORS.get(str(status), "gray")
            ax.scatter(
                group["longitude"], group["latitude"],
                s=10, label=status_dict[str(status)], color=color, alpha=0.9, **proj_kwargs
            )
            handles.append(Line2D([0], [0], marker="o", linestyle="", color=color, label=status_dict[str(status)]))
        

    # Optional path line through time (helps visualize motion). Sort by timestamp.
    ax.plot(
        df["longitude"], df["latitude"],
        linewidth=0.7, alpha=0.5, **proj_kwargs
    )

    # --- Plot plan CSVs (connected lines) ---
    if plan_dfs:
        from itertools import cycle
        plan_colors = cycle([
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
            "tab:olive", "tab:cyan"
        ])
        for pdf, label in plan_dfs:
            c = next(plan_colors)
            ax.plot(
                pdf["lon"], pdf["lat"],
                linewidth=1.8, alpha=0.9, label=f"plan: {label}", **proj_kwargs
            )
            # handles.append(Line2D([0], [0], color=c, lw=2, label=f"plan: {label}"))

    # Title
    title = "Plans with Signal Strength"
    if row_start is not None or row_end is not None:
        lo = row_start if row_start is not None else 1
        hi = row_end if row_end is not None else int(df_full["_orig_row"].max())
        # title += f" (rows {lo}–{hi})"
    ax.set_title(title)

    # Legend (deduplicate labels)
    seen = set()
    uniq_handles = []
    for h in handles:
        if h.get_label() not in seen:
            uniq_handles.append(h)
            seen.add(h.get_label())
    if uniq_handles:
        ax.legend(handles=uniq_handles, title="Legend", loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(img_name, dpi=200)
    plt.show()
    plt.close()
    return img_name


def main():
    parser = argparse.ArgumentParser(description="Plot GPS waypoints on a world map, with optional plan overlays.")
    parser.add_argument("fp", type=str, help="Filepath to .csv with columns: date,time,latitude,longitude,status")
    parser.add_argument("--row-start", type=int, default=None, help="1-based inclusive start row in waypoint CSV")
    parser.add_argument("--row-end", type=int, default=None, help="1-based inclusive end row in waypoint CSV")
    parser.add_argument("--img-name", type=str, default=None, help="Output image filename (defaults to <csv_basename>_waypoints.png)")
    parser.add_argument(
        "--plan-csv", type=str, nargs="+", default=None,
        help="One or more CSVs of plans (each row a plan; 'latlon_plan' column)."
    )
    args = parser.parse_args()

    out = plot_waypoints(args.fp, args.row_start, args.row_end, args.img_name, args.plan_csv)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
