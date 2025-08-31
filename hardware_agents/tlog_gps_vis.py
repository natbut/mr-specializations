#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot GPS lat/lon points from one or more CSV files on a satellite basemap.
Now supports multiple timestamp ranges, each drawn in a distinct color.

CSV requirement: columns 'gps.lat' and 'gps.lon' (case-sensitive here).
Optional time column for filtering; auto-detected if not specified.

Examples
--------
# Two time windows colored differently; duration-style offsets from earliest:
python tlog_gps_vis.py --csv-dir logs/ \
  --time-col Timestamp \
  --range "8/21/2025 11:38:01" "8/21/2025 11:52:01" \
  --range "49:00.0" "1:20:00.0" \
  --zoom 18 --yaml conf/lake_3.yaml

# Backward-compat single window (still supported):
python tlog_gps_vis.py file1.csv file2.csv \
  --time-col Timestamp --start "2025-08-21 11:38:01" --end "2025-08-21 11:52:01"

Notes
-----
- Basemap uses Esri.WorldImagery by default (internet required).
- Time boundaries accept either full datetimes or duration strings:
  "SSS", "MM:SS(.fff)", or "HH:MM:SS(.fff)". For duration on datetime
  columns, the offset is applied from the earliest timestamp present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import Point

# -------------------------- helpers: units check ---------------------------

def in_deg(a: pd.Series, b: pd.Series) -> bool:
    return a.between(-90, 90).all() and b.between(-180, 180).all()

def in_rad(a: pd.Series, b: pd.Series) -> bool:
    return a.between(-np.pi/2, np.pi/2).all() and b.between(-np.pi, np.pi).all()

def in_microdeg(a: pd.Series, b: pd.Series) -> bool:
    return (
        a.abs().median() > 90 and a.abs().median() < 1e7 and
        b.abs().median() > 180 and b.abs().median() < 1e7
    )

# -------------------- time parsing and CSV -> GeoDataFrame -----------------

def read_csv_to_gdf(
    csv_path: Path,
    time_col: str | None = None,
    t_start: str | None = None,
    t_end: str | None = None,
    time_mode: str = "auto",  # "auto" | "duration" | "datetime"
) -> gpd.GeoDataFrame:
    """Load CSV, optionally filter by time range, and return WGS84 GeoDataFrame."""

    # ----- helpers (duration parsing) -----
    def _norm_dur(s: str) -> str:
        s = str(s).strip().replace(",", ".")
        if ":" not in s and s.count(".") == 2:
            a, b, c = s.split(".")
            s = f"{a}:{b}.{c}"
        return s

    def _dur_str(s: str) -> bool:
        s = _norm_dur(s)
        return bool(
            # seconds
            (s.replace(".", "", 1).isdigit()) or
            # MM:SS(.fff) or HH:MM:SS(.fff)
            (":" in s and s.split(":")[0].isdigit())
        )

    def _parse_dur_sec(x: str) -> float:
        x = _norm_dur(x)
        if x.replace(".", "", 1).isdigit():
            return float(x)
        parts = x.split(":")
        if len(parts) == 2:  # MM:SS(.fff)
            m, s = parts
            return float(m) * 60.0 + float(s)
        if len(parts) == 3:  # HH:MM:SS(.fff)
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        raise ValueError(f"Unrecognized duration: {x!r}")

    def _series_is_duration(ser: pd.Series) -> bool:
        sample = ser.dropna().astype(str).head(50).map(str.strip)
        if sample.empty:
            return False
        hits = sum(_dur_str(x) for x in sample)
        return hits >= max(3, int(0.7 * len(sample)))

    # ----- load -----
    df = pd.read_csv(csv_path)
    missing = [c for c in ("gps.lat", "gps.lon") if c not in df.columns]
    if missing:
        raise ValueError(f"'{csv_path}' missing column(s): {', '.join(missing)}")

    # locate time column
    ts_col = None
    if time_col and time_col in df.columns:
        ts_col = time_col
    elif time_col:
        print(f"{csv_path.name}: specified --time-col '{time_col}' not found.", file=sys.stderr)
    else:
        cand = ["timestamp","time","gps.time","datetime","date","gpstime","utc_time","utc"]
        lmap = {c.lower(): c for c in df.columns}
        for k in cand:
            if k in lmap:
                ts_col = lmap[k]
                break

    # time filtering
    if (t_start is not None or t_end is not None) and ts_col is not None:
        ser = df[ts_col].astype(str).str.strip()

        # decide mode
        if time_mode == "duration":
            mode = "duration"
        elif time_mode == "datetime":
            mode = "datetime"
        else:
            mode = "duration" if _series_is_duration(ser) else "datetime"

        before = len(df)
        if mode == "duration":
            secs = ser.map(_parse_dur_sec)
            mask = pd.Series(True, index=df.index)
            if t_start is not None: mask &= secs >= _parse_dur_sec(str(t_start))
            if t_end   is not None: mask &= secs <= _parse_dur_sec(str(t_end))
            df = df.loc[mask].copy()
            print(f"{csv_path.name}: duration filter kept {len(df)}/{before} rows (col='{ts_col}').", file=sys.stderr)
        else:
            ts = pd.to_datetime(ser, errors="coerce", utc=True)
            t0 = ts.min()
            def _boundary(b):
                if b is None or b == "": return None
                b = str(b).strip()
                if _dur_str(b) and pd.notna(t0):
                    return t0 + pd.to_timedelta(_parse_dur_sec(b), unit="s")
                return pd.to_datetime(b, utc=True)
            start_dt = _boundary(t_start)
            end_dt   = _boundary(t_end)
            mask = ts.notna()
            if start_dt is not None: mask &= ts >= start_dt
            if end_dt   is not None: mask &= ts <= end_dt
            df = df.loc[mask].copy()
            print(f"{csv_path.name}: datetime filter kept {len(df)}/{before} rows (col='{ts_col}').", file=sys.stderr)
    elif (t_start is not None or t_end is not None) and ts_col is None:
        print(f"{csv_path.name}: no time column found; cannot apply time filter.", file=sys.stderr)

    # coordinate parsing / repair
    lat = pd.to_numeric(df["gps.lat"], errors="coerce")
    lon = pd.to_numeric(df["gps.lon"], errors="coerce")

    msg = []
    if in_deg(lat, lon):
        pass
    elif in_deg(lon, lat):
        msg.append("detected swapped lat/lon; auto-correcting"); lat, lon = lon, lat
    elif in_rad(lat, lon):
        msg.append("detected radians; converting to degrees"); lat = np.degrees(lat); lon = np.degrees(lon)
    elif in_rad(lon, lat):
        msg.append("detected radians+swapped; correcting"); lat, lon = np.degrees(lon), np.degrees(lat)
    elif in_microdeg(lat, lon):
        msg.append("detected microdegrees; dividing by 1e6"); lat = lat/1e6; lon = lon/1e6
    elif in_microdeg(lon, lat):
        msg.append("detected microdegrees+swapped; correcting"); lat, lon = lon/1e6, lat/1e6
    else:
        msg.append("values look unusual; assuming degrees as-is")
    if msg: print(f"{csv_path.name}: " + "; ".join(msg), file=sys.stderr)

    df["gps.lat"], df["gps.lon"] = lat, lon
    df = df.dropna(subset=["gps.lat","gps.lon"])
    zero_mask = (df["gps.lat"] == 0.0) | (df["gps.lon"] == 0.0)
    if zero_mask.any():
        print(f"{csv_path.name}: skipped {int(zero_mask.sum())} row(s) with 0.0 lat or lon.", file=sys.stderr)
    df = df.loc[~zero_mask].copy()

    # geometry
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["gps.lon"], df["gps.lat"]),
        crs="EPSG:4326",
    )
    return gdf

# ---------------------- YAML mothership/tasks extraction -------------------

def _is_number(x):
    try:
        float(x); return True
    except Exception:
        return False

def _valid_latlon(lat, lon):
    return (
        _is_number(lat) and _is_number(lon)
        and -90.0 <= float(lat) <= 90.0
        and -180.0 <= float(lon) <= 180.0
        and not (float(lat) == 0.0 and float(lon) == 0.0)
    )

def _iter_latlons(obj, lat_key="lat", lon_key="lon", path=()):
    if isinstance(obj, dict):
        if lat_key in obj and lon_key in obj and _valid_latlon(obj[lat_key], obj[lon_key]):
            yield (float(obj[lat_key]), float(obj[lon_key]), path)
        for k, v in obj.items():
            yield from _iter_latlons(v, lat_key, lon_key, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_latlons(v, lat_key, lon_key, path + (f"[{i}]",))

def load_yaml_points(
    yaml_path: Path,
    mother_key: str = "mothership_loc_latlon",
    tasks_key: str = "task_locs_latlon",
    lat_key: str = "lat",
    lon_key: str = "lon",
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    mother = None
    tasks = []

    if isinstance(data, dict):
        node = data.get(mother_key)
        if isinstance(node, (list, tuple)) and len(node) >= 2 and _valid_latlon(node[0], node[1]):
            mother = (float(node[0]), float(node[1]))
        if tasks_key in data and isinstance(data[tasks_key], list):
            for t in data[tasks_key]:
                if isinstance(t, (list, tuple)) and len(t) >= 2 and _valid_latlon(t[0], t[1]):
                    tasks.append((float(t[0]), float(t[1])))

    if mother is None or not tasks:
        mother_kw = ("mother", "mothership", "base", "home")
        tasks_kw = ("task", "tasks", "target", "targets", "waypoint", "waypoints", "goal", "goals")
        for lat, lon, path in _iter_latlons(data, lat_key, lon_key):
            pstr = "/".join(path).lower()
            if mother is None and any(k in pstr for k in mother_kw):
                mother = (lat, lon)
            elif any(k in pstr for k in tasks_kw):
                tasks.append((lat, lon))
        if not tasks:
            for lat, lon, path in _iter_latlons(data, lat_key, lon_key):
                if mother is None or (lat, lon) != mother:
                    tasks.append((lat, lon))

    gdf_mother = None
    if mother is not None:
        gdf_mother = gpd.GeoDataFrame(
            {"kind": ["mothership"]},
            geometry=gpd.points_from_xy([mother[1]], [mother[0]]),
            crs="EPSG:4326",
        )

    gdf_tasks = None
    if tasks:
        lats, lons = zip(*tasks)
        gdf_tasks = gpd.GeoDataFrame(
            {"idx": list(range(len(tasks)))},
            geometry=gpd.points_from_xy(lons, lats),
            crs="EPSG:4326",
        )

    return gdf_mother, gdf_tasks

# ---------------------------- QGC .plan parsing ----------------------------

def _best_latlon_from_vec(vec):
    if not isinstance(vec, (list, tuple)) or len(vec) < 2:
        return None
    cand = [(0,1), (1,0)]
    L = len(vec)
    cand += [(i,j) for i in range(L) for j in range(L) if i!=j and (i,j) not in cand]
    for i,j in cand:
        a,b = vec[i], vec[j]
        if _valid_latlon(a,b):
            return float(a), float(b)
    return None

def _extract_item_coords(item):
    pts = []
    if isinstance(item, dict) and "coordinate" in item:
        pair = _best_latlon_from_vec(item["coordinate"])
        if pair: pts.append(pair)
    if isinstance(item, dict) and isinstance(item.get("params"), list):
        vec = item["params"]
        if len(vec) >= 6:
            lat, lon = vec[4], vec[5]
            if _valid_latlon(lat, lon):
                pts.append((float(lat), float(lon)))
            elif _valid_latlon(lon, lat):
                pts.append((float(lon), float(lat)))
        if not pts:
            pair = _best_latlon_from_vec(vec)
            if pair: pts.append(pair)

    def _maybe_collect_path(container):
        if isinstance(container, dict) and "path" in container and isinstance(container["path"], list):
            for v in container["path"]:
                pair = _best_latlon_from_vec(v) if isinstance(v, (list, tuple)) else None
                if pair: pts.append(pair)
    for key in ("polyline", "polygon"):
        if key in item:
            _maybe_collect_path(item[key])
    for key in ("path", "points", "coordinates"):
        if key in item and isinstance(item[key], list):
            for v in item[key]:
                pair = _best_latlon_from_vec(v) if isinstance(v, (list, tuple)) else None
                if pair: pts.append(pair)
    cleaned, last = [], None
    for p in pts:
        if p != last:
            cleaned.append(p); last = p
    return cleaned

def parse_plan_file(plan_path: Path) -> tuple[str, gpd.GeoDataFrame | None]:
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to read {plan_path}: {e}", file=sys.stderr)
        return (plan_path.stem, None)

    mission = data.get("mission", {}) if isinstance(data, dict) else {}
    items = mission.get("items", []) if isinstance(mission, dict) else []
    coords = []
    for it in items:
        coords.extend(_extract_item_coords(it))

    cleaned, last = [], None
    for latlon in coords:
        if latlon != last:
            cleaned.append(latlon); last = latlon

    if not cleaned:
        print(f"Warning: no coordinates found in {plan_path}", file=sys.stderr)
        return (plan_path.stem, None)

    lats, lons = zip(*cleaned)
    gdf = gpd.GeoDataFrame(
        {"seq": list(range(len(cleaned)))},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    return (plan_path.stem, gdf)

def load_plan_dirs(plan_dirs: list[Path]) -> list[tuple[str, gpd.GeoDataFrame]]:
    out = []
    for d in plan_dirs or []:
        if not d.exists() or not d.is_dir():
            print(f"Warning: plan dir not found or not a dir: {d}", file=sys.stderr)
            continue
        files = sorted(Path(d).glob("*.plan"))
        if not files:
            print(f"Warning: no .plan files in {d}", file=sys.stderr)
        fplans = []
        for fp in files:
            label, gdf = parse_plan_file(fp)
            if gdf is not None and not gdf.empty:
                fplans.append((label, gdf))
        out.append(fplans)
    return out

# ------------------------ extent helpers & conversions ---------------------

def compute_mercator_bounds(gdfs_4326: List[gpd.GeoDataFrame], pad_ratio: float = 0.05) -> Tuple[float, float, float, float]:
    bounds = []
    for g in gdfs_4326:
        if g is None or g.empty:  # guard
            continue
        g_3857 = g.to_crs(epsg=3857)
        bounds.append(g_3857.total_bounds)  # (minx, miny, maxx, maxy)
    if not bounds:
        raise ValueError("No valid points found in any input file.")
    bounds = np.array(bounds)
    minx = bounds[:, 0].min()
    miny = bounds[:, 1].min()
    maxx = bounds[:, 2].max()
    maxy = bounds[:, 3].max()
    width = maxx - minx
    height = maxy - miny
    pad_x = width * pad_ratio if width > 0 else 1000
    pad_y = height * pad_ratio if height > 0 else 1000
    return minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y

def make_square_extent(extent_3857, pad_ratio=0.05):
    minx, miny, maxx, maxy = extent_3857
    width = maxx - minx
    height = maxy - miny
    side = max(width, height)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half = side / 2.0
    sq_minx, sq_maxx = cx - half, cx + half
    sq_miny, sq_maxy = cy - half, cy + half
    pad = side * pad_ratio
    return (sq_minx - pad, sq_miny - pad, sq_maxx + pad, sq_maxy + pad)

R = 6378137.0  # Web Mercator sphere radius (meters)
def x_to_lon(x):       return np.degrees(x / R)
def lon_to_x(lon):     return np.radians(lon) * R
def y_to_lat(y):       return np.degrees(np.arctan(np.sinh(y / R)))
def lat_to_y(lat):     return R * np.arcsinh(np.tan(np.radians(lat)))
def dd_fmt(x, _pos):   return f"{x:.4f}"

# --------------------------------- main -----------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot GPS lat/lon points on a satellite basemap with optional multi-range coloring.")
    # Inputs
    p.add_argument("--csv-dir", type=Path, default=None, help="Directory containing CSV files with 'gps.lat' and 'gps.lon'.")
    p.add_argument("csv_files", nargs="*", type=Path, help="(Deprecated) One or more CSV files with 'gps.lat' and 'gps.lon'.")

    # Time filtering
    p.add_argument("--time-col", type=str, default=None, help="Time column name (e.g., 'Timestamp').")
    p.add_argument("--start", type=str, default=None, help="Inclusive start (legacy single window).")
    p.add_argument("--end",   type=str, default=None, help="Inclusive end (legacy single window).")
    p.add_argument("--time-mode", choices=["auto", "duration", "datetime"], default="auto",
                   help="Force interpretation of the time column. Use 'duration' for MM:SS(.fff).")
    p.add_argument("--range", dest="ranges", action="append", nargs=2, metavar=("START","END"),
                   help="Add a timestamp range (inclusive). Repeat to add multiple. "
                        "Use '' for open-ended. Accepts duration strings or datetimes.")

    # Output & style
    p.add_argument("--out", type=Path, default=Path("gps_satellite_plot.png"), help="Output image file path.")
    p.add_argument("--point-size", type=float, default=10.0, help="Marker size for points.")
    p.add_argument("--alpha", type=float, default=0.9, help="Marker transparency (0..1).")
    p.add_argument("--connect", action="store_true", help="Draw dotted lines connecting points within each (file,range).")
    p.add_argument("--dpi", type=int, default=200, help="Output figure DPI.")
    p.add_argument("--title", type=str, default="Vehicle GPS Tracks", help="Figure title.")
    p.add_argument("--no-legend", action="store_true", help="Suppress legend display.")
    p.add_argument("--zoom", default="auto", help="Basemap zoom level (0-19 typical).")
    p.add_argument("--tile-url", type=str, default=None, help="Optional XYZ tile URL to override provider.")

    # Optional YAML points
    p.add_argument("--yaml", type=Path, default=None, help="YAML file with mothership and tasks lat/lon to plot.")
    p.add_argument("--yaml-mother-key", type=str, default="mothership_loc_latlon", help="YAML key for mothership lat/lon.")
    p.add_argument("--yaml-tasks-key", type=str, default="task_locs_latlon", help="YAML key for tasks list.")
    p.add_argument("--yaml-lat-key", type=str, default="lat", help="Latitude key name inside YAML objects.")
    p.add_argument("--yaml-lon-key", type=str, default="lon", help="Longitude key name inside YAML objects.")
    p.add_argument("--label-tasks", action="store_true", help="Annotate tasks with their index.")

    # Optional QGC plan overlays
    p.add_argument("--plan-dirs", nargs="+", type=Path, default=None, help="Directories containing QGroundControl .plan files.")

    args = p.parse_args()

    zoom_arg = str(args.zoom).strip().lower()
    zoom = 'auto' if zoom_arg in ("auto", "none", "default") else int(zoom_arg)

    # Collect CSV files
    if args.csv_dir is not None:
        if not args.csv_dir.exists() or not args.csv_dir.is_dir():
            print(f"Error: --csv-dir '{args.csv_dir}' does not exist or is not a directory.", file=sys.stderr)
            sys.exit(1)
        csv_files = sorted(args.csv_dir.glob("*.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in directory '{args.csv_dir}'.", file=sys.stderr)
            sys.exit(1)
    else:
        csv_files = list(args.csv_files)
        if not csv_files:
            print("Error: No CSV files specified.", file=sys.stderr)
            sys.exit(1)

    # Build list of ranges. If --range used, it overrides legacy --start/--end; otherwise use single legacy window.
    ranges: list[tuple[Optional[str], Optional[str]]] = []
    if args.ranges:
        for s, e in args.ranges:
            s = None if s is None or s == "" else s
            e = None if e is None or e == "" else e
            ranges.append((s, e))
    else:
        ranges.append((args.start, args.end))  # can be (None, None) => no filtering

    # Read CSVs per range
    all_ranges_gdfs: list[list[gpd.GeoDataFrame]] = []
    extent_inputs: list[gpd.GeoDataFrame] = []
    for (rstart, rend) in ranges:
        range_gdfs: list[gpd.GeoDataFrame] = []
        for pth in csv_files:
            try:
                gdf = read_csv_to_gdf(pth, time_col=args.time_col, t_start=rstart, t_end=rend, time_mode=args.time_mode)
                if not gdf.empty:
                    range_gdfs.append(gdf)
                    extent_inputs.append(gdf)
            except Exception as ex:
                print(f"Warning: failed reading {pth}: {ex}", file=sys.stderr)
        all_ranges_gdfs.append(range_gdfs)

    # Optional YAML points
    gdf_mother = gdf_tasks = None
    if args.yaml:
        try:
            gdf_mother, gdf_tasks = load_yaml_points(
                args.yaml,
                mother_key=args.yaml_mother_key,
                tasks_key=args.yaml_tasks_key,
                lat_key=args.yaml_lat_key,
                lon_key=args.yaml_lon_key,
            )
            if gdf_mother is not None and not gdf_mother.empty:
                extent_inputs.append(gdf_mother)
            if gdf_tasks is not None and not gdf_tasks.empty:
                extent_inputs.append(gdf_tasks)
        except Exception as e:
            print(f"Warning: failed to load YAML points from {args.yaml}: {e}", file=sys.stderr)

    # Optional plan overlays (extent not driven by plans to avoid skew)
    plan_tracks = []
    if args.plan_dirs:
        plan_tracks = load_plan_dirs(args.plan_dirs)

    if not extent_inputs:
        print("Error: nothing to plot after filters.", file=sys.stderr)
        sys.exit(1)

    # Figure & basemap
    extent_3857 = make_square_extent(compute_mercator_bounds(extent_inputs, pad_ratio=0.05), pad_ratio=0.05)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_xlim(extent_3857[0], extent_3857[2])
    ax.set_ylim(extent_3857[1], extent_3857[3])

    TILE_URL_DEFAULT = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    try:
        src = args.tile_url or TILE_URL_DEFAULT
        cx.add_basemap(ax, source=src, crs="EPSG:3857", zoom=zoom)
    except Exception as e:
        print(f"Warning: XYZ basemap failed ({e}). Trying provider...", file=sys.stderr)
        try:
            cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:3857", zoom=zoom)
        except Exception as e2:
            print(f"Warning: provider fallback failed ({e2}). Proceeding without background.", file=sys.stderr)

    # Draw YAML overlays
    if gdf_tasks is not None and not gdf_tasks.empty:
        g = gdf_tasks.to_crs(epsg=3857)
        g.plot(ax=ax, marker="o", markersize=max(200, args.point_size * 10),
               alpha=0.95, color="#ffd54f", edgecolor="black", linewidth=0.2,
               label=f"Tasks ({len(gdf_tasks)})")
        if args.label_tasks:
            for i, row in g.reset_index(drop=True).iterrows():
                ax.text(row.geometry.x, row.geometry.y, str(i), fontsize=8, ha="center", va="center")

    if gdf_mother is not None and not gdf_mother.empty:
        g = gdf_mother.to_crs(epsg=3857)
        g.plot(ax=ax, marker="*", markersize=max(300, args.point_size * 10),
               alpha=1.0, color="white", edgecolor="black", linewidth=0.8,
               label="Mothership")
        
    # Color per RANGE (key change from prior version)
    range_colors = ["white", "black"] #plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or [f"C{i}" for i in range(10)]

    for idx, (range_gdfs, (rstart, rend)) in enumerate(zip(all_ranges_gdfs, ranges)):
        if not range_gdfs: 
            continue
        color = range_colors[idx % len(range_colors)]
        label_txt = "All files" if (rstart is None and rend is None) else f"Vehicle {idx+1}"

        first = True
        for gdf in range_gdfs:
            g = gdf.to_crs(epsg=3857)
            # points
            g.plot(ax=ax, marker="o", markersize=args.point_size, alpha=args.alpha, color=color,
                   label=label_txt if first else None)
            # connect
            if args.connect and len(g) >= 2:
                xs = g.geometry.x.to_numpy(); ys = g.geometry.y.to_numpy()
                ax.plot(xs, ys, linewidth=2.0, alpha=min(args.alpha, 0.9), color=color) #, linestyle='dotted')
            first = False

    # Plans
    if plan_tracks:
        # Colors vary per *plan* within a folder; linestyles vary per *folder*
        plan_colors = sns.color_palette("pastel", n_colors=10)
        lines = ["solid", "dashed", "dotted", "dashdot"]

        for folder_idx, fplans in enumerate(plan_tracks):
            folder_linestyle = lines[folder_idx % len(lines)]
            for plan_idx, (label, gdf_wgs84_plan) in enumerate(fplans):
                color = plan_colors[plan_idx % len(plan_colors)]
                g = gdf_wgs84_plan.to_crs(epsg=3857).sort_values("seq")
                xs = g.geometry.x.to_numpy(); ys = g.geometry.y.to_numpy()
                ax.plot(xs, ys, linewidth=4.0, alpha=0.95,
                        color=color, linestyle=folder_linestyle,
                        label=f"Vehicle {folder_idx+1} Plan: {plan_idx+1}")
                ax.scatter(xs, ys, s=max(10, args.point_size), alpha=0.95, color=color,
                        edgecolors="white", linewidths=0.5)


    # Styling: axes with lat/lon secondaries
    ax.set_title(args.title, fontsize=28)

    secx = ax.secondary_xaxis('bottom', functions=(x_to_lon, lon_to_x))
    # secx.xaxis.label.set_size(13)
    secx.xaxis.set_tick_params(labelsize=12)
    secx.set_xlabel("Longitude (°)", fontsize=22)
    secy = ax.secondary_yaxis('left', functions=(y_to_lat, lat_to_y))
    # secx.yaxis.label.set_size(13)
    secy.yaxis.set_tick_params(labelsize=12)
    secy.set_ylabel("Latitude (°)", fontsize=22)
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    secx.xaxis.set_major_locator(MaxNLocator(nbins=6))
    secy.yaxis.set_major_locator(MaxNLocator(nbins=8))
    secx.xaxis.set_major_formatter(FuncFormatter(dd_fmt))
    secy.yaxis.set_major_formatter(FuncFormatter(dd_fmt))

    # Hide primary metric axes
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

    if not args.no_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True, fontsize=18)

    plt.tight_layout() #pad=0.5, h_pad=0.0)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved plot to: {out_path.resolve()}")
    plt.show()

if __name__ == "__main__":
    main()

# "C:/Users/Nathan Butler/Documents/OSU/RDML/mr-specializations/.venv/Scripts/python.exe" "c:/Users/Nathan Butler/Documents/OSU/RDML/mr-specializations/hardware_agents/tlog_gps_vis.py" --csv-dir hardware_agents\tlogs\2025-08-21  --zoom 19 --yaml hardware_agents\conf\lake_3.yaml --plan-dirs hardware_agents\lake3_2\plans_lake3_1_17_45_33 hardware_agents\lake3_2\plans_lake3_2_17_45_38  --time-col Timestamp --range "8/21/2025 11:39.30 AM" "8/21/2025 11:46:01 AM" --range "8/21/2025 12:18:01 PM" "8/21/2025 12:22:01 PM" --connect --point-size 5.0 --alpha 0.5 --title "Path Executions"


# "C:/Users/Nathan Butler/Documents/OSU/RDML/mr-specializations/.venv/Scripts/python.exe" "c:/Users/Nathan Butler/Documents/OSU/RDML/mr-specializations/hardware_agents/tlog_gps_vis.py" --csv-dir hardware_agents\tlogs\2025-08-21  --zoom 19 --yaml hardware_agents\conf\lake_3.yaml --plan-dirs hardware_agents\lake3_greedy\plans_lake3_1_18_39_58 hardware_agents\lake3_greedy\plans_lake3_2_18_40_05 --time-col Timestamp --range "8/21/2025 09:38.31 AM" "8/21/2025 09:44:31 AM" --range "8/21/2025 09:28:01 AM" "8/21/2025 09:34:31 AM" --connect --point-size 5.0 --alpha 0.5 --title "Path Executions"