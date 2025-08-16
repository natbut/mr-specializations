#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot GPS lat/lon points from one or more CSV files on a satellite basemap,
cropped to the region covered by the points. Each file is a unique color.

CSV schema requirement: columns named 'gps.lat' and 'gps.lon'.

Usage:
    python plot_gps_on_satellite.py path/to/file1.csv path/to/file2.csv \
        --out plot.png --point-size 12 --connect

Notes:
- Basemap source is Esri.WorldImagery (requires internet).
- If you see tile loading errors, re-run or try a smaller extent.
"""
import argparse
import json
import re
import sys
from glob import glob
from pathlib import Path
from typing import List, Tuple

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import Point


def in_deg(a, b):
        return a.between(-90, 90).all() and b.between(-180, 180).all()

def in_rad(a, b):
    return a.between(-np.pi/2, np.pi/2).all() and b.between(-np.pi, np.pi).all()

def in_microdeg(a, b):
    # very rough heuristic: typical absolute magnitudes >> degrees but << 1e9
    return a.abs().median() > 90 and a.abs().median() < 1e7 and \
            b.abs().median() > 180 and b.abs().median() < 1e7
            
def read_csv_to_gdf(
    csv_path: Path,
    time_col: str | None = None,
    t_start: str | None = None,
    t_end: str | None = None,
    time_mode: str = "auto",  # "auto" | "duration" | "datetime"
) -> gpd.GeoDataFrame:
    import re
    import sys

    import numpy as np
    import pandas as pd

    # ----- helpers (duration parsing) -----
    def _norm_dur(s: str) -> str:
        s = str(s).strip().replace(",", ".")
        # fix "MM.SS.sss" -> "MM:SS.sss"
        if ":" not in s and s.count(".") == 2:
            a, b, c = s.split(".")
            s = f"{a}:{b}.{c}"
        return s

    def _dur_str(s: str) -> bool:
        s = _norm_dur(s)
        return bool(re.fullmatch(r"\d+(?:\.\d+)?", s) or
                    re.fullmatch(r"\d{1,3}:\d{2}(?:\.\d+)?", s) or
                    re.fullmatch(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", s))

    def _parse_dur_sec(x: str) -> float:
        x = _norm_dur(x)
        if re.fullmatch(r"\d+(?:\.\d+)?", x):  # seconds
            return float(x)
        parts = x.split(":")
        if len(parts) == 2:                    # MM:SS(.fff)
            m, s = parts;  return float(m)*60.0 + float(s)
        if len(parts) == 3:                    # HH:MM:SS(.fff)
            h, m, s = parts;  return float(h)*3600.0 + float(m)*60.0 + float(s)
        raise ValueError(f"Unrecognized duration: {x!r}")

    def _series_is_duration(ser: pd.Series) -> bool:
        sample = ser.dropna().astype(str).head(50).map(str.strip)
        if sample.empty: return False
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
            if k in lmap: ts_col = lmap[k]; break

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
                if b is None: return None
                b = str(b).strip()
                # If boundary looks like duration (e.g., '49:00.1'), treat as offset from earliest ts
                if _dur_str(b) and pd.notna(t0):
                    return t0 + pd.to_timedelta(_parse_dur_sec(b), unit="s")
                # Otherwise try full datetime
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

    # coordinate parsing / repair (unchanged)
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

    # write back, drop NaNs & zeros
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

def _is_valid_latlon(lat, lon) -> bool:
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

def _iter_latlons(obj, lat_key="lat", lon_key="lon", path=()):
    """Yield (lat, lon, path_tokens) for any dicts containing lat_key/lon_key."""
    if isinstance(obj, dict):
        if lat_key in obj and lon_key in obj and _is_valid_latlon(obj[lat_key], obj[lon_key]):
            yield (float(obj[lat_key]), float(obj[lon_key]), path)
        for k, v in obj.items():
            yield from _iter_latlons(v, lat_key, lon_key, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_latlons(v, lat_key, lon_key, path + (f"[{i}]",))

def load_yaml_points(
    yaml_path: Path,
    mother_key: str = "mothership",
    tasks_key: str = "tasks",
    lat_key: str = "lat",
    lon_key: str = "lon",
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    """
    Returns (gdf_mother_wgs84, gdf_tasks_wgs84). Either may be None/empty.
    Primary strategy: use explicit keys; fallback: recursive search with heuristics.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        
    mother = None
    tasks = []

    # 1) Direct schema
    if isinstance(data, dict):
        print("Mother data:", data.get(mother_key))
        node = data.get(mother_key)
        if _is_valid_latlon(node[0], node[1]):
            mother = (float(node[0]), float(node[1]))

        print("Task data:", data[tasks_key])
        if tasks_key in data and isinstance(data[tasks_key], list):
            for t in data[tasks_key]:
                if _is_valid_latlon(t[0], t[1]):
                    tasks.append((float(t[0]), float(t[1])))

    # 2) Fallback: scan everything
    if mother is None or not tasks:
        mother_kw = ("mother", "mothership", "base", "home")
        tasks_kw = ("task", "tasks", "target", "targets", "waypoint", "waypoints", "goal", "goals")

        for lat, lon, path in _iter_latlons(data, lat_key, lon_key):
            pstr = "/".join(path).lower()
            if mother is None and any(k in pstr for k in mother_kw):
                mother = (lat, lon)
            elif any(k in pstr for k in tasks_kw):
                tasks.append((lat, lon))

        # If still nothing tagged as tasks, but many points exist, treat all non-mother points as tasks
        if not tasks:
            for lat, lon, path in _iter_latlons(data, lat_key, lon_key):
                if mother is None or (lat, lon) != mother:
                    tasks.append((lat, lon))

    # Build GeoDataFrames in WGS84
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



def compute_mercator_bounds(gdfs_4326: List[gpd.GeoDataFrame], pad_ratio: float = 0.05) -> Tuple[float, float, float, float]:
    """Compute padded bounds in EPSG:3857 covering all input GeoDataFrames."""
    # Reproject each to Web Mercator and collect bounds
    bounds = []
    for g in gdfs_4326:
        if g.empty:
            continue
        g_3857 = g.to_crs(epsg=3857)
        bounds.append(g_3857.total_bounds)  # (minx, miny, maxx, maxy)
    if not bounds:
        raise ValueError("No valid points found in any input file.")

    import numpy as np
    bounds = np.array(bounds)
    minx = bounds[:, 0].min()
    miny = bounds[:, 1].min()
    maxx = bounds[:, 2].max()
    maxy = bounds[:, 3].max()

    # Pad bounds a bit for nicer framing
    width = maxx - minx
    height = maxy - miny
    pad_x = width * pad_ratio if width > 0 else 1000  # meters
    pad_y = height * pad_ratio if height > 0 else 1000

    return minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y


def make_square_extent(extent_3857, pad_ratio=0.05):
    """
    Given (minx, miny, maxx, maxy) in EPSG:3857, pad to a square extent
    using the longer side. Also adds a small outer padding.
    """
    minx, miny, maxx, maxy = extent_3857
    width = maxx - minx
    height = maxy - miny
    side = max(width, height)

    # center current extent
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    half = side / 2.0
    sq_minx, sq_maxx = cx - half, cx + half
    sq_miny, sq_maxy = cy - half, cy + half

    # final padding
    pad = side * pad_ratio
    return (sq_minx - pad, sq_miny - pad, sq_maxx + pad, sq_maxy + pad)

def _parse_dt(s):
    return pd.to_datetime(s, utc=True) if s else None



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

def _best_latlon_from_vec(vec):
    """
    Given a list/tuple like [lat, lon, alt] (or sometimes [lon, lat, alt]),
    return a (lat, lon) tuple if any adjacent pair looks valid.
    Preference order: (0,1), then (1,0), else scan all pairs.
    """
    if not isinstance(vec, (list, tuple)) or len(vec) < 2:
        return None
    # prefer first two
    cand = [(0,1), (1,0)]
    # add all other pairs if present
    L = len(vec)
    cand += [(i,j) for i in range(L) for j in range(L) if i!=j and (i,j) not in cand]
    for i,j in cand:
        a,b = vec[i], vec[j]
        if _valid_latlon(a,b):
            return float(a), float(b)
    return None

def _extract_item_coords(item):
    """
    Extract ordered (lat, lon) points from a single QGC mission item.
    Handles:
      - 'coordinate': [lat, lon, alt]
      - 'params': [..., lat, lon, alt]  (indices 4,5 are typical for NAV cmds)
      - 'polyline'/'polygon': {'path': [[lat, lon, alt], ...]}
      - generic 'path' / 'points' / 'coordinates' lists
    """
    pts = []

    # 1) direct coordinate (common in some exports)
    if isinstance(item, dict) and "coordinate" in item:
        coord = item["coordinate"]
        if isinstance(coord, (list, tuple)):
            pair = _best_latlon_from_vec(coord)
            if pair: pts.append(pair)

    # 2) params array (QGC SimpleItem with MAV_CMD)
    if isinstance(item, dict) and isinstance(item.get("params"), list):
        vec = item["params"]
        # Typical: [..., lat, lon, alt] starting at index 4
        if len(vec) >= 6:
            lat, lon = vec[4], vec[5]
            if _valid_latlon(lat, lon):
                pts.append((float(lat), float(lon)))
            elif _valid_latlon(lon, lat):  # some tools store lon,lat
                pts.append((float(lon), float(lat)))
        # Fallback: scan any adjacent pair in params
        if not pts:
            pair = _best_latlon_from_vec(vec)
            if pair: pts.append(pair)

    # 3) polyline/polygon containers with 'path'
    def _maybe_collect_path(container):
        if isinstance(container, dict) and "path" in container and isinstance(container["path"], list):
            for v in container["path"]:
                pair = _best_latlon_from_vec(v) if isinstance(v, (list, tuple)) else None
                if pair: pts.append(pair)

    for key in ("polyline", "polygon"):
        if key in item:
            _maybe_collect_path(item[key])

    # 4) generic lists
    for key in ("path", "points", "coordinates"):
        if key in item and isinstance(item[key], list):
            for v in item[key]:
                pair = _best_latlon_from_vec(v) if isinstance(v, (list, tuple)) else None
                if pair: pts.append(pair)

    # De-dup consecutive duplicates while preserving order
    cleaned, last = [], None
    for p in pts:
        if p != last:
            cleaned.append(p); last = p
    return cleaned


def parse_plan_file(plan_path: Path) -> tuple[str, gpd.GeoDataFrame | None]:
    """
    Parse a QGroundControl .plan file and return (label, GeoDataFrame WGS84) or (label, None)
    The GDF has columns: ['seq'] and geometry (points), ordered by mission sequence.
    """
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to read {plan_path}: {e}", file=sys.stderr)
        return (plan_path.stem, None)

    items = []
    if isinstance(data, dict):
        mission = data.get("mission", {})
        if isinstance(mission, dict) and isinstance(mission.get("items"), list):
            items = mission["items"]

    coords = []
    for it in items:
        coords.extend(_extract_item_coords(it))

    # Deduplicate consecutive duplicates while preserving order
    cleaned = []
    last = None
    for latlon in coords:
        if latlon != last:
            cleaned.append(latlon)
            last = latlon

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
    """
    Scan directories for *.plan and parse each.
    Returns list of (label, GeoDataFrame) for plans that yielded points.
    """
    out = []
    for d in plan_dirs or []:
        if not d.exists() or not d.is_dir():
            print(f"Warning: plan dir not found or not a dir: {d}", file=sys.stderr)
            continue
        files = sorted(Path(d).glob("*.plan"))
        if not files:
            print(f"Warning: no .plan files in {d}", file=sys.stderr)
        for fp in files:
            label, gdf = parse_plan_file(fp)
            if gdf is not None and not gdf.empty:
                out.append((label, gdf))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot GPS lat/lon points from CSV(s) on a satellite basemap."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="One or more CSV files with 'gps.lat' and 'gps.lon' columns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gps_satellite_plot.png"),
        help="Output image file path (e.g., plot.png). Default: gps_satellite_plot.png",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=10.0,
        help="Marker size for points. Default: 10",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Marker transparency (0..1). Default: 0.9",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="If set, draw lines connecting points in each file (in CSV order).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure DPI. Default: 200",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="GPS Points on Satellite Basemap",
        help="Figure title.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress legend display.",
    )
    parser.add_argument(
        "--zoom", default='auto',
        help="Basemap zoom level (0-19 typical). Set explicitly to avoid provider zoom metadata issues."
    )
    parser.add_argument(
        "--tile-url", type=str, default=None,
        help="Optional XYZ tile URL (e.g., Esri World Imagery). If set, overrides provider."
    )
    parser.add_argument("--time-col", type=str, default=None,
        help="Time column name (e.g., 'Timestamp').")
    parser.add_argument("--start", type=str, default=None,
        help="Inclusive start. For duration columns use 'MM:SS(.fff)' or seconds. For datetime columns, pass ISO or use a duration to mean 'offset from earliest'.")
    parser.add_argument("--end", type=str, default=None,
        help="Inclusive end (same rules as --start).")
    parser.add_argument("--time-mode", choices=["auto", "duration", "datetime"], default="auto",
        help="Force interpretation of the time column. Use 'duration' for MM:SS(.fff).")
    parser.add_argument(
        "--yaml", type=Path, default=None,
        help="YAML file with mothership and tasks lat/lon to plot."
    )
    parser.add_argument(
        "--yaml-mother-key", type=str, default="mothership_loc_latlon",
        help="Key under which the mothership lat/lon lives (default: 'mothership')."
    )
    parser.add_argument(
        "--yaml-tasks-key", type=str, default="task_locs_latlon",
        help="Key under which the tasks list lives (default: 'tasks')."
    )
    parser.add_argument(
        "--yaml-lat-key", type=str, default="lat",
        help="Latitude key name inside YAML objects (default: 'lat')."
    )
    parser.add_argument(
        "--yaml-lon-key", type=str, default="lon",
        help="Longitude key name inside YAML objects (default: 'lon')."
    )
    parser.add_argument(
        "--label-tasks", action="store_true",
        help="If set, annotate tasks with their index."
    )
    parser.add_argument(
        "--plan-dirs", nargs="+", type=Path, default=None,
        help="One or more directories containing QGroundControl .plan files to plot."
    )
        
    args = parser.parse_args()
    
    zoom_arg = str(args.zoom).strip().lower()
    zoom = 'auto' if zoom_arg in ("auto", "none", "default") else int(zoom_arg)

    # t_start = _parse_dt(args.start)
    # t_end   = _parse_dt(args.end)
    
    
    # Read all CSVs to GeoDataFrames (WGS84)
    print("Reading CSVs...")
    gdfs_wgs84 = []
    labels = []
    for p in args.csv_files:
        if not p.exists():
            print(f"Warning: '{p}' does not exist. Skipping.", file=sys.stderr)
            continue
        gdf = read_csv_to_gdf(
                p,
                time_col=args.time_col,
                t_start=args.start,
                t_end=args.end,
                time_mode=args.time_mode,
        )
        if gdf.empty:
            print(f"Warning: '{p}' has no valid rows after cleaning. Skipping.", file=sys.stderr)
            continue
        gdfs_wgs84.append(gdf)
        labels.append(p.stem)
        
    # --- Optional YAML points ---
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
            if gdf_mother is not None:
                gdfs_wgs84.append(gdf_mother)
                labels.append("mothership")
            if gdf_tasks is not None and len(gdf_tasks):
                gdfs_wgs84.append(gdf_tasks)
                labels.append("tasks")
            print(f"Loaded YAML: mother={0 if gdf_mother is None else 1}, tasks={0 if gdf_tasks is None else len(gdf_tasks)}")
        except Exception as e:
            print(f"Warning: failed to load YAML points from {args.yaml}: {e}", file=sys.stderr)


    if not gdfs_wgs84:
        print("Error: No valid input files.", file=sys.stderr)
        sys.exit(1)
        
    # for lbl, g in zip(labels, gdfs_wgs84):
    #     print(f"Min/max: {lbl}: lat[{g['gps.lat'].min():.6f}, {g['gps.lat'].max():.6f}] "
    #         f"lon[{g['gps.lon'].min():.6f}, {g['gps.lon'].max():.6f}]")


    # --- Load QGC plan paths (optional) ---
    plan_tracks = []
    if args.plan_dirs:
        plan_tracks = load_plan_dirs(args.plan_dirs)

    # Build list of layers to drive extent (CSV + YAML + plan tracks)
    extent_inputs = list(gdfs_wgs84)  # CSV datasets

    if 'gdf_tasks' in locals() and gdf_tasks is not None and len(gdf_tasks):
        extent_inputs.append(gdf_tasks)
    if 'gdf_mother' in locals() and gdf_mother is not None and len(gdf_mother):
        extent_inputs.append(gdf_mother)
    for _, gdf in plan_tracks:
        extent_inputs.append(gdf)

    if not extent_inputs:
        print("Error: nothing to plot after filters.", file=sys.stderr)
        sys.exit(1)

    # Compute bounds in Web Mercator to set extent & draw basemap
    print("Computing bounds...")
    extent_3857 = compute_mercator_bounds(extent_inputs, pad_ratio=0.05)
    extent_3857 = make_square_extent(extent_3857, pad_ratio=0.05)

    # Prepare figure
    print("Preparing figure...")
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_xlim(extent_3857[0], extent_3857[2])
    ax.set_ylim(extent_3857[1], extent_3857[3])

    # Add basemap first using the desired extent
    # Set the axis to the computed extent before adding basemap so it's cropped
    ax.set_xlim(extent_3857[0], extent_3857[2])
    ax.set_ylim(extent_3857[1], extent_3857[3])

    # Esri World Imagery
    print("Adding basemap...")
    TILE_URL_DEFAULT = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )

    basemap_ok = False
    try:
        src = args.tile_url or TILE_URL_DEFAULT  # prefer raw URL
        cx.add_basemap(ax, source=src, crs="EPSG:3857", zoom=zoom)
        basemap_ok = True
    except Exception as e:
        print(f"Warning: XYZ basemap failed ({e}). Trying provider...", file=sys.stderr)
        try:
            cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:3857", zoom=zoom)
            basemap_ok = True
        except Exception as e2:
            print(f"Warning: provider fallback failed ({e2}). Proceeding without background.", file=sys.stderr)

    # --- Plot YAML points if present ---
    if gdf_tasks is not None and len(gdf_tasks):
        gdf_tasks_3857 = gdf_tasks.to_crs(epsg=3857)
        gdf_tasks_3857.plot(
            ax=ax,
            marker="o",
            markersize=max(200, args.point_size * 10),
            alpha=0.95,
            color="#ffd54f",        # amber-ish for contrast
            edgecolor="black",
            linewidth=0.2,
            label=f"Tasks ({len(gdf_tasks)})",
        )

        if args.label_tasks:
            for i, row in gdf_tasks_3857.reset_index(drop=True).iterrows():
                ax.text(row.geometry.x, row.geometry.y, str(i), fontsize=8, ha="center", va="center")

    if gdf_mother is not None and len(gdf_mother):
        gdf_mother_3857 = gdf_mother.to_crs(epsg=3857)
        gdf_mother_3857.plot(
            ax=ax,
            marker="*",
            markersize=max(200, args.point_size * 10),
            alpha=1.0,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            label="Mothership",
        )

    # --- Plot QGC plan tracks (always connected) ---
    if plan_tracks:
        plan_color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not plan_color_cycle:
            plan_color_cycle = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

        for i, (label, gdf_wgs84_plan) in enumerate(plan_tracks):
            color = plan_color_cycle[i % len(plan_color_cycle)]
            gdf_3857_plan = gdf_wgs84_plan.to_crs(epsg=3857).sort_values("seq")

            # connect in order
            xs = gdf_3857_plan.geometry.x.to_numpy()
            ys = gdf_3857_plan.geometry.y.to_numpy()
            ax.plot(xs, ys, linewidth=2.0, alpha=0.95, color=color, label=f"Plan: {label}")

            # waypoints as small markers
            ax.scatter(xs, ys, s=max(10, args.point_size), alpha=0.95, color=color, edgecolors="white", linewidths=0.5)


    # Color cycle (enough for many files; will wrap if exceeded)
    print("Plotting points...")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    print("Color cycle:", color_cycle)
    if not color_cycle:
        color_cycle = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    cmap_cycle = ["Blues", "Oranges", "Greens", "pink", ]

    # Plot each dataset (CSV tracks only should connect)
    for i, (gdf_wgs84, label) in enumerate(zip(gdfs_wgs84, labels)):
        
        color = color_cycle[i % len(color_cycle)]
        color_map = cmap_cycle[i % len(cmap_cycle)]
        # gdf_3857_plan = gdf_wgs84_plan.to_crs(epsg=3857).sort_values("seq")
        
        # Create a gradient colormap (e.g., viridis) based on point order
        cmap = plt.get_cmap(color_map)
        n_points = len(gdf_wgs84)
        colors = [cmap(j / max(n_points - 1, 1)) for j in range(n_points)]
        gdf_3857 = gdf_wgs84.to_crs(epsg=3857)
        

        # Points
        gdf_3857.plot(
            ax=ax,
            marker="o",
            markersize=args.point_size,
            alpha=args.alpha,
            color=colors,
            label=label,
        )

        # ⬇️ add this guard so tasks/mothership never connect
        if (
            args.connect
            and len(gdf_3857) >= 2
            and str(label).lower() not in {"tasks", "mothership", "plan"}
        ):
            xs = gdf_3857.geometry.x.to_numpy()
            ys = gdf_3857.geometry.y.to_numpy()
            ax.plot(xs, ys, linewidth=1.5, alpha=min(args.alpha, 0.9), color=color)
        elif (
            args.connect
            and len(gdf_3857) >= 2
            and str(label).lower() in {"plan"}
        ):
            xs = gdf_3857.geometry.x.to_numpy()
            ys = gdf_3857.geometry.y.to_numpy()
            ax.plot(xs, ys, linewidth=1.5, alpha=min(args.alpha, 0.9), color=color_cycle[-1])

    # Styling
    ax.set_title(args.title, fontsize=14)
    ax.set_xlabel("Web Mercator X (meters)")
    ax.set_ylabel("Web Mercator Y (meters)")
    
    if not args.no_legend:
        ax.legend(loc='center right', bbox_to_anchor=(0.0, 0.5), frameon=True)
        
    secx = ax.secondary_xaxis('top', functions=(x_to_lon, lon_to_x))
    secx.set_xlabel("Longitude (deg)")

    secy = ax.secondary_yaxis('right', functions=(y_to_lat, lat_to_y))
    secy.set_ylabel("Latitude (deg)")
    
    secx.xaxis.set_major_locator(MaxNLocator(nbins=6))
    secy.yaxis.set_major_locator(MaxNLocator(nbins=8))
    secx.xaxis.set_major_formatter(FuncFormatter(dd_fmt))
    secy.yaxis.set_major_formatter(FuncFormatter(dd_fmt))

    secx.set_xlabel("Longitude (°)")
    secy.set_ylabel("Latitude (°)")

    # Hide primary meter axes (ticks and labels)
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Tight layout and save
    plt.tight_layout()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved plot to: {out_path.resolve()}")
    
    plt.show()



R = 6378137.0  # Web Mercator sphere radius (meters)

def x_to_lon(x):       # meters -> degrees
    return np.degrees(x / R)

def lon_to_x(lon):     # degrees -> meters
    return np.radians(lon) * R

def y_to_lat(y):       # meters -> degrees
    return np.degrees(np.arctan(np.sinh(y / R)))

def lat_to_y(lat):     # degrees -> meters
    return R * np.arcsinh(np.tan(np.radians(lat)))

# Clean decimal-degree formatting
def dd_fmt(x, _pos):
    # adaptive: show 5 decimals for sub-meter at equator, fewer for wide extents
    return f"{x:.5f}"



if __name__ == "__main__":
    main()


# "c:/Users/Nathan Butler/Documents/OSU/RDML/mr-specializations/hardware_agents/tlog_gps_vis.py" hardware_agents\tlogs\2025-08-13_15-47-44_vehicle1.csv hardware_agents\tlogs\2025-08-13_09-48-43_vehicle1.csv hardware_agents\tlogs\2025-08-13_11-51-09_vehicle1.csv hardware_agents\tlogs\2025-08-13_09-45-59_vehicle2.csv hardware_agents\tlogs\2025-08-13_11-51-10_vehicle2.csv hardware_agents\tlogs\2025-08-13_15-55-24_vehicle2.csv  hardware_agents\tlogs\2025-08-13_15-47-44_vehicle2.csv --zoom 19 --yaml hardware_agents/conf/lake_1.yaml --plan-dirs hardware_agents\logs_1150am\_1 hardware_agents\logs_1150am\_2 --time-col Timestamp --start "8/13/2025 11:40.01 AM" --end "8/13/2025 1:30:01 PM"

# "c:/Users/Nathan Butler/Documents/OSU/RDML/mr-specializations/hardware_agents/tlog_gps_vis.py" hardware_agents\tlogs\2025-08-13_11-51-09_vehicle1.csv hardware_agents\tlogs\2025-08-13_11-51-10_vehicle2.csv --zoom 19 --yaml hardware_agents/conf/lake_1.yaml --plan-dirs hardware_agents\logs_1150am\_1 hardware_agents\logs_1150am\_2 --time-col Timestamp --start "8/13/2025 11:40.01 AM" --end "8/13/2025 1:30:01 PM"

# "c:/Users/Nathan Butler/Documents/OSU/RDML/mr-specializations/hardware_agents/tlog_gps_vis.py" hardware_agents\tlogs\2025-08-13_15-47-44_vehicle1.csv hardware_agents\tlogs\2025-08-13_09-48-43_vehicle1.csv hardware_agents\tlogs\2025-08-13_11-51-09_vehicle1.csv --zoom 19 --yaml hardware_agents/conf/lake_1.yaml --plan-dirs hardware_agents\logs_1150am\_1 --time-col Timestamp --start "8/13/2025 11:40.01 AM" --end "8/13/2025 1:30:01 PM"