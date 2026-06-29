"""
Load and gap-fill single-station in-situ forcing for ShellSIM.

This is the in-situ counterpart to ``extract_filled_grid.py``.  The satellite
pipeline gap-fills cloud / missing-day gaps on a ``(time, lat, lon)`` cube with
**DINEOF** — an EOF/PCA reconstruction that exploits the *spatial* covariance
between pixels.  An in-situ mooring is a SINGLE point: there is no spatial
dimension for DINEOF to operate on, so the faithful 1-D analogue of the
satellite gap-fill is **temporal interpolation onto the same daily horizon**.

Pipeline (per variable) — mirrors ``extract_filled_grid._to_reference_grid``
followed by the DINEOF fill, but in 1-D:

1. Read the measured series from the Excel workbook.
2. Collapse duplicate / same-day samples to one daily value
   (``resample('1D').mean()`` — identical to the satellite daily step).
3. Reindex onto the full daily horizon (``pd.date_range(freq='D')``), leaving
   missing days as NaN (identical to the satellite ``reindex`` step).
4. Fill the NaN gaps by temporal interpolation (default linear in time), then
   forward/back-fill the ends with a flat hold (no wild extrapolation) — the
   1-D stand-in for DINEOF's reconstruction.

Returns a daily ``xarray.Dataset`` with variables ``temperature``, ``salinity``
and ``Chl`` on a ``time`` axis (plus scalar ``latitude`` / ``longitude``
coords), so it drops straight into the same ShellSIM wrapper the gridded run
uses.  Also returns the raw measured points (for plotting) and a coverage dict.

In-situ source (Culture Project, IVL, Gullmarsfjord 2019-2021), 3 m depth:
    sheet 'CPr food Chl 3m' -> Chl (mg m-3)
    sheet 'CPr Env 3m'      -> T (degC), S (psu)
"""

import numpy as np
import pandas as pd
import xarray as xr

CHL_SHEET = "CPr food Chl 3m"
ENV_SHEET = "CPr Env 3m"
FOOD_SHEET = "CPr food TPM POM 3m"   # POC (unmeasured -> zeros), POM, TPM


def _daily_reindex(dates, values, time_horizon):
    """Daily-mean collapse + reindex onto the daily horizon (gaps stay NaN).

    Mirrors the satellite ``resample('1D').mean()`` then ``reindex(time)``
    steps in ``extract_filled_grid._to_reference_grid`` — no fill yet.
    """
    s = pd.Series(np.asarray(values, dtype=float),
                  index=pd.to_datetime(dates)).sort_index()
    s = s[~s.index.isna()]
    s = s.resample("1D").mean()        # collapse same-day samples to one value
    s = s.reindex(time_horizon)        # missing days preserved as NaN
    return s


def _fill_temporal(s, method="linear"):
    """1-D analogue of DINEOF: interpolate over time, then hold the ends.

    After ``_daily_reindex`` the index is uniform (1 day), so ``linear`` and
    ``time`` are equivalent.  Leading/trailing gaps (before the first / after
    the last measurement) cannot be interpolated, so they are held flat with
    ffill/bfill rather than extrapolated.
    """
    filled = s.interpolate(method=method, limit_direction="both")
    filled = filled.ffill().bfill()
    return filled


def load_filled_insitu(excel_file, time_range, latitude=None, longitude=None,
                       interp_method="linear", clip_negative=True, verbose=False):
    """Read in-situ T/S/Chl, gap-fill onto a daily horizon, return a Dataset.

    Parameters
    ----------
    excel_file : str
        Path to 'Environmental Data for PML.xlsx'.
    time_range : (str, str)
        (start, end) dates for the daily horizon, e.g. ('2020-02-01','2021-08-31').
    latitude, longitude : float, optional
        Station coordinates, attached as scalar coords for record-keeping.
    interp_method : str
        pandas interpolation method for the gap fill (default 'linear').
    clip_negative : bool
        Clamp filled Chl to >= 0 (negative concentrations are unphysical and
        diverge ShellSIM's forward-Euler step).
    verbose : bool
        Print horizon + per-variable measured-day coverage.

    Returns
    -------
    ds : xr.Dataset
        Daily ``temperature``, ``salinity``, ``Chl`` on the ``time`` axis.
    raw : dict[str, pd.DataFrame]
        Raw measured points keyed 'Chl','T','S' (columns: date, value).
    coverage : dict[str, int]
        Number of measured days landing on the horizon, per variable.
    """
    start, end = time_range
    time_horizon = pd.date_range(start=start, end=end, freq="D")

    # --- read raw measurements ---
    chl_df = pd.read_excel(excel_file, sheet_name=CHL_SHEET).rename(columns={"!date": "date"})
    chl_df["date"] = pd.to_datetime(chl_df["date"])
    chl_df = chl_df[["date", "Chl"]].dropna()

    env_df = pd.read_excel(excel_file, sheet_name=ENV_SHEET).rename(columns={"!date": "date"})
    env_df["date"] = pd.to_datetime(env_df["date"])
    env_df = env_df[["date", "T", "S"]].dropna()

    raw = {
        "Chl": chl_df.rename(columns={"Chl": "value"})[["date", "value"]],
        "T":   env_df[["date", "T"]].rename(columns={"T": "value"}),
        "S":   env_df[["date", "S"]].rename(columns={"S": "value"}),
    }

    # --- daily reindex (gaps = NaN), then temporal fill ---
    chl_daily = _daily_reindex(chl_df["date"], chl_df["Chl"], time_horizon)
    t_daily = _daily_reindex(env_df["date"], env_df["T"], time_horizon)
    s_daily = _daily_reindex(env_df["date"], env_df["S"], time_horizon)

    coverage = {"Chl": int(chl_daily.notna().sum()),
                "T": int(t_daily.notna().sum()),
                "S": int(s_daily.notna().sum())}

    chl_f = _fill_temporal(chl_daily, interp_method)
    t_f = _fill_temporal(t_daily, interp_method)
    s_f = _fill_temporal(s_daily, interp_method)

    if clip_negative:
        chl_f = chl_f.clip(lower=0.0)

    ds = xr.Dataset(
        {
            "temperature": ("time", t_f.values.astype("float32")),
            "salinity":    ("time", s_f.values.astype("float32")),
            "Chl":         ("time", chl_f.values.astype("float32")),
        },
        coords={"time": time_horizon},
    )
    if latitude is not None:
        ds = ds.assign_coords(latitude=float(latitude))
    if longitude is not None:
        ds = ds.assign_coords(longitude=float(longitude))

    ds["temperature"].attrs = {"units": "degC", "long_name": "In-situ temperature (3 m)"}
    ds["salinity"].attrs = {"units": "psu", "long_name": "In-situ salinity (3 m)"}
    ds["Chl"].attrs = {"units": "mg m-3", "long_name": "In-situ chlorophyll (3 m)"}

    if verbose:
        print(f"Daily horizon: {time_horizon[0].date()} -> {time_horizon[-1].date()} "
              f"({len(time_horizon)} days)")
        for k, n in coverage.items():
            print(f"  {k:3s}: {n:4d} measured days on horizon -> filled to {len(time_horizon)}")

    return ds, raw, coverage


def load_filled_insitu_full(excel_file, time_range, latitude=None, longitude=None,
                            interp_method="linear", clip_negative=True,
                            food_unit_factor=1.0, verbose=False):
    """Like :func:`load_filled_insitu` but ALSO reads POC / POM / TPM.

    POC / POM / TPM come from the ``FOOD_SHEET`` ('CPr food TPM POM 3m') and are
    gap-filled with the same 1-D temporal pipeline (daily collapse -> reindex ->
    interpolate -> hold ends).

    Units
    -----
    ShellSIM prey expect POC in mg m-3 (= ug/L) and POM / TPM in g m-3 (= mg/L).
    The workbook's POM (~0.5-3.5) and TPM (~4.6-13.8) magnitudes are only
    physical as mg/L (= g m-3), so they are used directly
    (``food_unit_factor=1.0``).  Override ``food_unit_factor`` if your column is
    truly in g/L (would be x1000).  **POC is NOT measured in the Culture Project
    (column is all zeros)** — that food channel is therefore effectively off
    even when full_fabm.yaml couples it.

    Returns
    -------
    ds : xr.Dataset      daily temperature, salinity, Chl, POC, POM, TPM
    raw : dict           raw measured points keyed Chl,T,S,POC,POM,TPM
    coverage : dict      measured days on the horizon per variable
    """
    # Base T / S / Chl via the standard loader.
    ds, raw, coverage = load_filled_insitu(
        excel_file, time_range, latitude=latitude, longitude=longitude,
        interp_method=interp_method, clip_negative=clip_negative, verbose=False)

    time_horizon = pd.to_datetime(ds["time"].values)

    food = pd.read_excel(excel_file, sheet_name=FOOD_SHEET).rename(columns={"!date": "date"})
    food["date"] = pd.to_datetime(food["date"])
    food = food[["date", "POC", "POM", "TPM"]].dropna()

    for col in ("POC", "POM", "TPM"):
        raw[col] = food[["date", col]].rename(columns={col: "value"})
        daily = _daily_reindex(food["date"], food[col], time_horizon)
        coverage[col] = int(daily.notna().sum())
        filled = _fill_temporal(daily, interp_method) * food_unit_factor
        if clip_negative:
            filled = filled.clip(lower=0.0)
        ds[col] = ("time", filled.values.astype("float32"))

    ds["POC"].attrs = {"units": "mg m-3", "long_name": "In-situ POC (3 m) [UNMEASURED - zeros]"}
    ds["POM"].attrs = {"units": "g m-3 (mg/L)", "long_name": "In-situ POM (3 m)"}
    ds["TPM"].attrs = {"units": "g m-3 (mg/L)", "long_name": "In-situ TPM (3 m)"}

    if verbose:
        print(f"Daily horizon: {time_horizon[0].date()} -> {time_horizon[-1].date()} "
              f"({len(time_horizon)} days)")
        for k in ("T", "S", "Chl", "POC", "POM", "TPM"):
            print(f"  {k:3s}: {coverage[k]:4d} measured days on horizon -> filled to {len(time_horizon)}")
        if (raw["POC"]["value"] == 0).all():
            print("  NOTE: POC is all zeros (not measured) -> POC food channel is effectively off.")

    return ds, raw, coverage
