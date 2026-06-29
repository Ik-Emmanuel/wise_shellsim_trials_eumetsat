"""
Extract and gap-fill a multi-variable ocean dataset from the Open Data Cube.

This module combines two approaches:

* the **DINEOF** gap-filling reconstruction (EOF/PCA-based, with cross-validated
  mode selection) — the more rigorous fill for cloud/day gaps, and
* the **per-variable land masking** that restores land / cloud-only pixels to
  NaN so the coastline shows correctly on maps.

It returns a daily ``(time, latitude, longitude)`` cube with variables
``salinity``, ``temperature``, ``chlorophyll`` and ``ocean_mask``.

Method (per variable)
----------------------
1. Load the native product (cmems_sss, s3_slstr_sst, cmems_chl_tur).
2. Collapse sub-daily / irregular scenes to one daily value per pixel
   (``resample('1D').mean()``).
3. Bilinearly regrid onto a common reference grid (``resolution`` degrees, or
   the native CHL grid if ``resolution=None``) and reindex onto the full daily
   horizon, **preserving gaps as NaN** for DINEOF to reconstruct.
4. Build a land/sea mask from the high-resolution products (SST | CHL pixels
   with >=1 real observation).
5. Gap-fill each variable with DINEOF over the ocean mask.
6. Restore land / cloud-only pixels of SST and CHL to NaN (``where(valid)``) so
   the coastline shows — except salinity, whose native grid (cmems_sss, 0.125
   deg) is too coarse to mask, so it is extrapolated and left unmasked
   (smooth gradient over the whole frame).

Products (default_resolution)
------------------------------
    cmems_sss      EPSG:4326  0.125   (L4 multi-sensor SSS, daily, gap-free)
    s3_slstr_sst   EPSG:4326  0.01    (Sentinel-3 SLSTR L2 SST swaths)
    cmems_chl_tur  EPSG:4326  0.001   (CMEMS NWS BGC high-res L3 NRT daily CHL)

Note on resolution: DINEOF reconstructs a ``(time, nlat*nlon)`` matrix with
PCA, so cost grows with pixel count.  The cmems_chl_tur native grid (0.001 deg)
is far too fine for DINEOF over any sizeable bbox — pass a coarser
``resolution`` (e.g. 0.003) to keep it tractable while still resolving the
coastline.

Usage
-----
As a module:
    from extract_filled_grid import load_filled_grid
    ds = load_filled_grid(
        bbox=[10.706177, 56.857983, 12.683716, 58.788132],
        time_range=('2021-05-01', '2021-09-01'),
        resolution=0.003,
        verbose=True,
    )

As a script:
    python extract_filled_grid.py \
        --bbox 10.706177 56.857983 12.683716 58.788132 \
        --start 2021-05-01 --end 2021-09-01 [--resolution 0.003] [--verbose]
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

# Products / measurements.
SSS_PRODUCT, SSS_VAR = "cmems_sss", "sos"
SST_PRODUCT, SST_VAR = "s3_slstr_sst", "sea_surface_temperature"
CHL_PRODUCT, CHL_VAR = "cmems_chl_tur", "CHL"


# ---------------------------------------------------------------------------
# DINEOF helpers
# ---------------------------------------------------------------------------


def determine_optimal_modes(data_ocean, valid_mask, cv_mask, cv_values, max_modes=20):
    """Choose the number of EOF modes that minimises cross-validation RMSE."""
    if cv_values is None or len(cv_values) < 10:
        return min(10, max_modes)

    nt, npts = data_ocean.shape
    modes_to_test = list(range(1, min(max_modes + 1, nt, npts), 2))
    if max_modes not in modes_to_test:
        modes_to_test.append(max_modes)
    modes_to_test = sorted(modes_to_test)[:15]

    cv_errors = []
    reconstructed = None
    for n in modes_to_test:
        filled = data_ocean.copy()
        temporal_means = np.nanmean(filled, axis=0)
        for j in range(filled.shape[1]):
            mask = np.isnan(filled[:, j])
            filled[mask, j] = temporal_means[j]
        global_mean = np.nanmean(data_ocean)
        filled[np.isnan(filled)] = global_mean
        filled[cv_mask] = np.nan

        for _ in range(10):
            tmean = np.nanmean(filled, axis=0, keepdims=True)
            centered = filled - tmean
            cf = centered.copy()
            for j in range(cf.shape[1]):
                cm = np.nanmean(cf[:, j])
                cf[np.isnan(cf[:, j]), j] = 0.0 if np.isnan(cm) else cm
            try:
                pca = PCA(n_components=n)
                tc = pca.fit_transform(cf)
                reconstructed = tc @ pca.components_ + tmean
                mask_fill = ~valid_mask | cv_mask
                filled[mask_fill] = reconstructed[mask_fill]
            except Exception:
                break

        if reconstructed is not None:
            cv_errors.append(
                np.sqrt(np.mean((reconstructed[cv_mask] - cv_values) ** 2))
            )
        else:
            cv_errors.append(np.inf)

    return modes_to_test[int(np.argmin(cv_errors))]


def dineof_reconstruction(
    data_var,
    ocean_mask_da,
    n_modes="auto",
    max_iterations=100,
    tolerance=1e-4,
    cross_validation_fraction=0.05,
):
    """
    DINEOF gap-filling for a single xarray.DataArray.

    Parameters
    ----------
    data_var : xr.DataArray  (time, latitude, longitude)
    ocean_mask_da : xr.DataArray  bool, (latitude, longitude), True = ocean
    n_modes : int or 'auto'
    max_iterations : int
    tolerance : float
    cross_validation_fraction : float

    Returns
    -------
    xr.DataArray  — same shape/coords as data_var, gaps filled over ocean
    """
    log.info("  Processing %s...", data_var.name)

    # Ensure canonical dim order for the reshape below.
    data_var = data_var.transpose("time", "latitude", "longitude")
    ocean_mask_da = ocean_mask_da.transpose("latitude", "longitude")

    nt = len(data_var.time)
    nlat = len(data_var.latitude)
    nlon = len(data_var.longitude)

    data_matrix = data_var.values.reshape(nt, nlat * nlon)
    ocean_flat = ocean_mask_da.values.flatten()
    ocean_indices = np.where(ocean_flat)[0]
    data_ocean = data_matrix[:, ocean_indices]

    log.debug("    Data shape: %d time steps x %d ocean points", nt, len(ocean_indices))

    valid_mask = ~np.isnan(data_ocean)
    n_valid = valid_mask.sum()
    log.debug(
        "    Initial coverage: %d/%d (%.2f%%)",
        n_valid,
        data_ocean.size,
        n_valid / data_ocean.size * 100 if data_ocean.size else 0.0,
    )

    # Cross-validation setup
    cv_mask = np.zeros_like(valid_mask, dtype=bool)
    valid_indices = np.argwhere(valid_mask)
    n_cv = int(len(valid_indices) * cross_validation_fraction)
    cv_values = None

    if n_cv > 0:
        cv_sel = np.random.choice(len(valid_indices), size=n_cv, replace=False)
        cv_pos = valid_indices[cv_sel]
        cv_mask[cv_pos[:, 0], cv_pos[:, 1]] = True
        cv_values = data_ocean[cv_mask].copy()
        log.debug(
            "    Cross-validation: %d points withheld (%.1f%%)",
            n_cv,
            cross_validation_fraction * 100,
        )

    if n_modes == "auto":
        log.debug("    Determining optimal number of EOF modes...")
        n_modes = determine_optimal_modes(
            data_ocean,
            valid_mask,
            cv_mask,
            cv_values,
            max_modes=min(20, nt - 1, len(ocean_indices) - 1),
        )
        log.info("    Optimal number of modes: %d", n_modes)
    else:
        log.debug("    Using %d EOF modes", n_modes)

    # Initial fill: temporal mean per location, then global mean
    filled = data_ocean.copy()
    temporal_means = np.nanmean(filled, axis=0)
    for j in range(filled.shape[1]):
        miss = np.isnan(filled[:, j])
        filled[miss, j] = temporal_means[j]
    global_mean = np.nanmean(data_ocean)
    filled[np.isnan(filled)] = global_mean

    if cv_values is not None:
        filled[cv_mask] = np.nan

    log.debug("    Starting DINEOF iterations (max: %d)...", max_iterations)
    prev_error = np.inf
    prev_reconstruction = None

    for iteration in range(max_iterations):
        tmean = np.nanmean(filled, axis=0, keepdims=True)
        centered = filled - tmean
        cf = centered.copy()
        for j in range(cf.shape[1]):
            cm = np.nanmean(cf[:, j])
            cf[np.isnan(cf[:, j]), j] = 0.0 if np.isnan(cm) else cm

        try:
            pca = PCA(n_components=n_modes)
            tc = pca.fit_transform(cf)
            reconstructed = tc @ pca.components_ + tmean
        except Exception as e:
            log.error("    PCA failed at iteration %d: %s", iteration, e)
            break

        mask_fill = ~valid_mask | cv_mask
        filled[mask_fill] = reconstructed[mask_fill]

        if cv_values is not None:
            cv_error = np.sqrt(np.mean((reconstructed[cv_mask] - cv_values) ** 2))
            error_change = abs(prev_error - cv_error)
            if (iteration + 1) % 10 == 0:
                log.debug(
                    "      Iteration %d: CV-RMSE = %.6f, Change = %.6f",
                    iteration + 1,
                    cv_error,
                    error_change,
                )
            if error_change < tolerance:
                log.info(
                    "    Converged at iteration %d (CV-RMSE = %.6f)",
                    iteration + 1,
                    cv_error,
                )
                break
            prev_error = cv_error
        else:
            if prev_reconstruction is not None:
                change = np.sqrt(
                    np.mean(
                        (reconstructed[~valid_mask] - prev_reconstruction[~valid_mask])
                        ** 2
                    )
                )
                if (iteration + 1) % 10 == 0:
                    log.debug(
                        "      Iteration %d: Reconstruction change = %.6f",
                        iteration + 1,
                        change,
                    )
                if change < tolerance:
                    log.info("    Converged at iteration %d", iteration + 1)
                    break
            prev_reconstruction = reconstructed.copy()

    if cv_values is not None:
        filled[cv_mask] = cv_values

    # Reconstruct full spatial field
    filled_full = np.full((nt, nlat * nlon), np.nan)
    filled_full[:, ocean_indices] = filled
    filled_data = filled_full.reshape(nt, nlat, nlon)

    result = xr.DataArray(
        filled_data,
        coords=data_var.coords,
        dims=data_var.dims,
        attrs=data_var.attrs,
    )

    ocean_3d = ocean_mask_da.values[np.newaxis, :, :]
    valid_ocean = (~np.isnan(result.values)) & ocean_3d
    denom = ocean_3d.sum() * nt
    final_cov = valid_ocean.sum() / denom * 100 if denom else 0.0
    log.info("    Final coverage (ocean only): %.2f%%", final_cov)
    return result


# ---------------------------------------------------------------------------
# Gridding helpers
# ---------------------------------------------------------------------------


def _to_reference_grid(
    da, ref_lats, ref_lons, time_horizon, kelvin_to_celsius=False, extrapolate=False
):
    """Put a (time, lat, lon) DataArray on the reference grid + daily horizon.

    1) collapse sub-daily / irregular scenes to one daily value per pixel,
    2) bilinearly regrid onto the reference lat/lon grid
       (``extrapolate=False`` keeps NaN outside the data footprint so land
       stays NaN and can be masked later),
    3) reindex onto the full daily horizon — **gaps stay NaN** so DINEOF (not a
       straight-line interpolation) reconstructs the missing days.
    """
    da = da.resample(time="1D").mean()
    if kelvin_to_celsius:
        da = da - 273.15
    kwargs = {"fill_value": "extrapolate"} if extrapolate else {}
    da = da.interp(latitude=ref_lats, longitude=ref_lons, kwargs=kwargs)
    da = da.reindex(time=time_horizon)
    return da


def _fallback_fill(da, valid_2d):
    """Fill any residual NaNs within the valid mask: temporal mean, then mean.

    DINEOF normally fills every ocean pixel, but this guards against pixels it
    left empty (e.g. a column with no observations at all).
    """
    vals = da.values.copy()
    valid_3d = np.broadcast_to(valid_2d[np.newaxis, :, :], vals.shape)
    residual = np.isnan(vals) & valid_3d
    if not residual.any():
        return da
    log.warning("    Fallback fill for %s (%d pixels)", da.name, int(residual.sum()))
    tmean = np.nanmean(vals, axis=0)
    vals[residual] = np.broadcast_to(tmean, vals.shape)[residual]
    still = np.isnan(vals) & valid_3d
    if still.any():
        vals[still] = np.nanmean(vals[valid_3d])
    return xr.DataArray(vals, coords=da.coords, dims=da.dims, attrs=da.attrs)


def _build_target_grid(chl, bbox, resolution):
    """Reference lat/lon for regridding.

    Default (``resolution=None``): the native CHL grid.  WARNING — the
    cmems_chl_tur native grid is 0.001 deg, which is far too fine for DINEOF
    over a sizeable bbox; pass an explicit ``resolution`` (e.g. 0.003).  If a
    *resolution* (degrees) is given, build a regular grid at that spacing over
    the bbox, preserving the native latitude orientation.
    """
    ref_lats = chl["latitude"]
    ref_lons = chl["longitude"]

    if resolution is None:
        return ref_lats, ref_lons

    west, south, east, north = bbox
    lats = np.arange(south, north + resolution, resolution)
    lons = np.arange(west, east + resolution, resolution)
    # Preserve the native latitude direction (products are usually descending).
    if float(ref_lats[0]) > float(ref_lats[-1]):
        lats = lats[::-1]

    ref_lats = xr.DataArray(lats, dims="latitude", coords={"latitude": lats})
    ref_lons = xr.DataArray(lons, dims="longitude", coords={"longitude": lons})
    return ref_lats, ref_lons


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def load_filled_grid(
    bbox,
    time_range,
    resolution=0.003,
    dask_chunks=None,
    n_modes="auto",
    max_iterations=100,
    tolerance=1e-4,
    cross_validation_fraction=0.05,
    random_seed=42,
    mask_land=True,
    verbose=False,
    **_legacy_ignored,
):
    """
    Load SSS, SST, and CHL from the ODC, align them to a common daily grid,
    gap-fill each variable with DINEOF, and land-mask SST/CHL.

    Parameters
    ----------
    bbox : list[float]
        [west, south, east, north] in degrees (WGS-84).
    time_range : tuple[str, str]
        Start and end dates, e.g. ('2021-05-01', '2021-09-01').
    resolution : float or None
        Output grid spacing in degrees (default 0.003).  ``None`` uses the
        native CHL grid (0.001 deg) — NOT recommended for DINEOF over a
        sizeable bbox (the PCA matrix becomes huge).
    dask_chunks : dict or None
        Dask chunk sizes for ``dc.load``.  Default ``{'time': -1}``.
    n_modes : int or 'auto'
        Number of EOF modes for DINEOF.  'auto' selects via cross-validation.
    max_iterations : int
        Maximum DINEOF iterations per variable.
    tolerance : float
        Convergence tolerance for DINEOF.
    cross_validation_fraction : float
        Fraction of valid pixels withheld for cross-validation.
    random_seed : int
        Random seed for reproducibility.
    mask_land : bool
        If True, restore land / cloud-only pixels of SST and CHL to NaN so the
        coastline shows.  Salinity is always left unmasked + extrapolated: its
        native grid (cmems_sss, 0.125 deg) is too coarse for a land mask.
    verbose : bool
        If True, emit DEBUG-level progress.

    Returns
    -------
    xr.Dataset
        Variables: ``salinity``, ``temperature``, ``chlorophyll``,
        ``ocean_mask``.  Temperature in degC; land pixels NaN (white on maps).
    """
    import warnings
    import datacube
    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    if verbose:
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            log.addHandler(logging.StreamHandler())
    logging.getLogger("datacube").setLevel(logging.ERROR)  # make datacube less pink

    if _legacy_ignored:
        log.debug("Ignoring legacy kwargs: %s", sorted(_legacy_ignored))

    np.random.seed(random_seed)

    if dask_chunks is None:
        dask_chunks = {"time": -1}

    west, south, east, north = bbox
    dc = datacube.Datacube()

    load_kwargs = dict(
        lon=(west, east),
        lat=(south, north),
        time=time_range,
        dask_chunks=dask_chunks,
    )

    _t_load = time.time()

    log.info("Loading SSS (%s)...", SSS_PRODUCT)
    _t = time.time()
    sss_data = dc.load(product=SSS_PRODUCT, measurements=[SSS_VAR], **load_kwargs)
    log.debug("  SSS loaded in %.1fs", time.time() - _t)

    log.info("Loading SST (%s)...", SST_PRODUCT)
    _t = time.time()
    sst_data = dc.load(product=SST_PRODUCT, measurements=[SST_VAR], **load_kwargs)
    log.debug("  SST loaded in %.1fs", time.time() - _t)

    log.info("Loading CHL (%s)...", CHL_PRODUCT)
    _t = time.time()
    chl_data = dc.load(product=CHL_PRODUCT, measurements=[CHL_VAR], **load_kwargs)
    log.debug("  CHL loaded in %.1fs", time.time() - _t)

    # Guard: each product must return some data for the bbox/time_range.
    for name, ds_, var in (
        ("SSS", sss_data, SSS_VAR),
        ("SST", sst_data, SST_VAR),
        ("CHL", chl_data, CHL_VAR),
    ):
        if var not in ds_ or ds_[var].size == 0:
            raise ValueError(
                f"No {name} data returned for the given bbox and time_range. "
                "Check that they overlap indexed products."
            )

    # CHL (cmems_chl_tur) is the reference product for the grid.
    ref_lats, ref_lons = _build_target_grid(chl_data, bbox, resolution)
    log.info(
        "Reference grid: %d lat x %d lon%s",
        ref_lats.size,
        ref_lons.size,
        "" if resolution is None else f" @ {resolution} deg",
    )

    time_horizon = pd.date_range(start=time_range[0], end=time_range[1], freq="D")
    log.info(
        "Daily horizon: %s -> %s (%d days)",
        time_horizon[0].date(),
        time_horizon[-1].date(),
        len(time_horizon),
    )

    log.info("Regridding each variable (gaps preserved as NaN)...")
    # Salinity: native grid too coarse to mask -> extrapolate, leave unmasked.
    sss_grid = _to_reference_grid(
        sss_data[SSS_VAR], ref_lats, ref_lons, time_horizon, extrapolate=True
    )
    # Temperature: Kelvin -> degC.
    sst_grid = _to_reference_grid(
        sst_data[SST_VAR], ref_lats, ref_lons, time_horizon, kelvin_to_celsius=True
    )
    chl_grid = _to_reference_grid(chl_data[CHL_VAR], ref_lats, ref_lons, time_horizon)

    # Per-variable sea masks (pixels with >=1 real observation over the period).
    sal_valid = sss_grid.notnull().any("time")
    temp_valid = sst_grid.notnull().any("time")
    chl_valid = chl_grid.notnull().any("time")
    # Land/sea mask from the high-resolution products (SST | CHL observations).
    ocean_mask = (temp_valid | chl_valid).rename("ocean_mask")

    # Materialise the lazy dask arrays before DINEOF (it operates on numpy).
    log.debug("Computing regridded arrays...")
    _t = time.time()
    sss_grid = sss_grid.compute()
    sst_grid = sst_grid.compute()
    chl_grid = chl_grid.compute()
    sal_valid = sal_valid.compute()
    temp_valid = temp_valid.compute()
    chl_valid = chl_valid.compute()
    ocean_mask = ocean_mask.compute()
    log.debug("  Compute done in %.1fs", time.time() - _t)

    dineof_kwargs = dict(
        n_modes=n_modes,
        max_iterations=max_iterations,
        tolerance=tolerance,
        cross_validation_fraction=cross_validation_fraction,
    )

    log.info("Starting DINEOF gap-filling...")
    t0 = time.time()

    # Salinity over its own (full, extrapolated) mask; left unmasked.
    salinity_grid = dineof_reconstruction(
        sss_grid.rename("salinity"), sal_valid, **dineof_kwargs
    )
    salinity_grid = _fallback_fill(salinity_grid, sal_valid.values)

    # Temperature / chlorophyll over the shared ocean mask, then land-masked.
    temperature_grid = dineof_reconstruction(
        sst_grid.rename("temperature"), ocean_mask, **dineof_kwargs
    )
    temperature_grid = _fallback_fill(temperature_grid, ocean_mask.values)

    chl_filled = dineof_reconstruction(
        chl_grid.rename("chlorophyll"), ocean_mask, **dineof_kwargs
    )
    chl_filled = _fallback_fill(chl_filled, ocean_mask.values)

    log.info("Total DINEOF time: %.2f min", (time.time() - t0) / 60)

    if mask_land:
        temperature_grid = temperature_grid.where(temp_valid)
        chl_filled = chl_filled.where(chl_valid)

    result = xr.Dataset(
        {
            "salinity": salinity_grid.astype("float32"),
            "temperature": temperature_grid.astype("float32"),
            "chlorophyll": chl_filled.astype("float32"),
            "ocean_mask": ocean_mask,
        }
    )

    result["salinity"].attrs = {"units": "psu", "long_name": "Sea surface salinity"}
    result["temperature"].attrs = {
        "units": "degC",
        "long_name": "Sea surface temperature",
    }
    result["chlorophyll"].attrs = {
        "units": "mg m-3",
        "long_name": "Chlorophyll concentration",
    }

    log.info("Done in %.1fs", time.time() - _t_load)
    log.debug(
        "  NaNs per variable (land/empty x days): %s",
        {
            v: int(result[v].isnull().sum())
            for v in ("salinity", "temperature", "chlorophyll")
        },
    )
    return result


# ---------------------------------------------------------------------------
# Visualisation (optional convenience)
# ---------------------------------------------------------------------------


def plot_timestep(dataset, selected_date, return_fig=True):
    """Plot salinity, temperature, and chlorophyll at a single date.

    Parameters
    ----------
    dataset : xr.Dataset
        Output of :func:`load_filled_grid`.
    selected_date : str | datetime-like
        Date to plot; the nearest available time step is used.
    """
    import matplotlib.pyplot as plt

    selected_dt = np.datetime64(str(selected_date), "ns")
    time_index = int(np.argmin(np.abs(dataset.time.values - selected_dt)))
    date_label = str(dataset.time.values[time_index])[:10]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    dataset["salinity"].isel(time=time_index).plot(
        ax=axes[0], cmap="viridis", cbar_kwargs={"label": "Salinity (PSU)"}
    )
    axes[0].set_title(f"Sea Surface Salinity\n{date_label}")

    dataset["temperature"].isel(time=time_index).plot(
        ax=axes[1], cmap="coolwarm", cbar_kwargs={"label": "Temperature (degC)"}
    )
    axes[1].set_title(f"Sea Surface Temperature\n{date_label}")

    dataset["chlorophyll"].isel(time=time_index).plot(
        ax=axes[2], cmap="YlGn", cbar_kwargs={"label": "Chlorophyll (mg/m3)"}
    )
    axes[2].set_title(f"Chlorophyll\n{date_label}")

    plt.show()
    if return_fig:
        return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Load and DINEOF gap-fill SSS/SST/CHL from the ODC."
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
        help="Bounding box in WGS-84 degrees",
    )
    p.add_argument("--start", required=True, metavar="DATE", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, metavar="DATE", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--resolution",
        type=float,
        default=0.003,
        help="Grid spacing in degrees (default: 0.003)",
    )
    p.add_argument(
        "--modes", default="auto", help="Number of EOF modes, or 'auto' (default: auto)"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--no-mask-land",
        action="store_true",
        help="Do not restore land pixels to NaN",
    )
    p.add_argument("--output", metavar="FILE", help="Optional NetCDF output path")
    p.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    n_modes = args.modes if args.modes == "auto" else int(args.modes)

    combined_data = load_filled_grid(
        bbox=args.bbox,
        time_range=(args.start, args.end),
        resolution=args.resolution,
        n_modes=n_modes,
        random_seed=args.seed,
        mask_land=not args.no_mask_land,
        verbose=args.verbose,
    )

    log.info("Result:\n%s", combined_data)

    if args.output:
        combined_data.to_netcdf(args.output)
        log.info("Saved to %s", args.output)
