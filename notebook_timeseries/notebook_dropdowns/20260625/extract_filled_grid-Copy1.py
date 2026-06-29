"""
Extract and gap-fill a multi-variable ocean dataset from the Open Data Cube.

This module reproduces the validated gridding/gap-filling method from
``gridded_timeseries_datacube.ipynb`` (the "BUILD A PROPER GRIDDED, GAP-FILLED
COMBINED DATASET" cell).  It returns a daily (time, latitude, longitude) cube
with proper per-variable land masking.

Method (per variable)
----------------------
1. Load the native product (no forced reprojection).
2. Collapse sub-daily / irregular scenes to one daily value per pixel
   (``resample('1D').mean()``).
3. Bilinearly regrid onto the high-resolution CHL (s3_olci_chl) reference grid.
4. Interpolate per pixel, linearly in time, onto the full daily horizon
   (this fills internal day gaps while preserving spatial structure).
5. Build a land/sea mask from the pixels that have at least one real
   observation (``notnull().any('time')``).
6. Fill any days/pixels still missing with a domain-mean fallback series,
   then the overall mean.
7. Restore land / cloud-only pixels to NaN (``where(valid)``) so the coastline
   shows correctly — except salinity, whose native grid (~2x3 pixels) is too
   coarse to mask, so it is extrapolated and left unmasked (smooth gradient).

Usage
-----
As a module:
    from extract_filled_grid import load_filled_grid
    ds = load_filled_grid(
        bbox=[10.706177, 56.857983, 12.683716, 58.788132],
        time_range=('2021-05-01', '2021-09-01'),
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

log = logging.getLogger(__name__)

# Products / measurements (same as the validated notebook).
SSS_PRODUCT, SSS_VAR = "cmems_sss", "sos"
SST_PRODUCT, SST_VAR = "s3_slstr_sst", "sea_surface_temperature"
CHL_PRODUCT, CHL_VAR = "s3_olci_chl", "CHL_NN"


# ---------------------------------------------------------------------------
# Gridding / gap-filling helpers
# ---------------------------------------------------------------------------


def _to_reference_grid(
    da, ref_lats, ref_lons, time_horizon, kelvin_to_celsius=False, extrapolate=False
):
    """Put a (time, lat, lon) DataArray on the reference grid + daily horizon.

    1) collapse sub-daily / irregular scenes to one daily value per pixel,
    2) bilinearly regrid onto the reference lat/lon grid
       (``extrapolate=False`` keeps NaN outside the data footprint so land
       stays NaN and can be masked later),
    3) interpolate per pixel, linearly in time, onto the full daily horizon
       (fills internal day gaps while preserving spatial structure).
    """
    da = da.resample(time="1D").mean()
    if kelvin_to_celsius:
        da = da - 273.15
    kwargs = {"fill_value": "extrapolate"} if extrapolate else {}
    da = da.interp(latitude=ref_lats, longitude=ref_lons, kwargs=kwargs)
    da = da.interp(time=time_horizon)
    return da


def _series_to_horizon(grid, time_horizon):
    """Domain-mean daily fallback series aligned to the full horizon (no NaN).

    Used only to fill days where *every* pixel is missing (e.g. long cloud
    gaps or edges outside the observed date range).
    """
    spatial_dims = [d for d in grid.dims if d != "time"]
    series = grid.mean(dim=spatial_dims)
    series = series.reindex(time=time_horizon)
    series = series.interpolate_na("time")  # internal gaps (scipy linear)
    return series.fillna(series.mean())  # leading/trailing edges -> series mean


def _gridded_filled(
    raw_da,
    ref_lats,
    ref_lons,
    time_horizon,
    kelvin_to_celsius=False,
    mask_land=True,
    extrapolate=False,
):
    """Regrid, gap-fill, and (optionally) land-mask one variable.

    Returns
    -------
    (grid, valid) : (xr.DataArray, xr.DataArray)
        ``grid``  — filled (time, lat, lon) field, float32.
        ``valid`` — 2-D bool sea mask (pixels with >=1 real observation).
    """
    grid = _to_reference_grid(
        raw_da,
        ref_lats,
        ref_lons,
        time_horizon,
        kelvin_to_celsius=kelvin_to_celsius,
        extrapolate=extrapolate,
    )

    # sea mask = pixels that have at least one real observation over the period.
    # land / always-empty pixels are False -> restored to NaN at the end.
    valid = grid.notnull().any("time")

    fill = _series_to_horizon(grid, time_horizon)  # 1-D (time,) fallback
    grid = grid.fillna(fill)  # fill missing DAYS on observed pixels
    grid = grid.fillna(float(grid.mean()))  # any pixel still empty -> overall mean

    if mask_land:
        grid = grid.where(valid)  # put land/cloud-only pixels back to NaN (white)

    return grid.astype("float32"), valid


def _build_target_grid(chl, bbox, resolution):
    """Reference lat/lon for regridding.

    Default (``resolution=None``): the native s3_olci_chl grid — the highest
    resolution in the collection, which resolves the coastline.  If a
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
    resolution=None,
    dask_chunks=None,
    mask_land=True,
    verbose=False,
    **_legacy_ignored,
):
    """
    Load SSS, SST, and CHL from the ODC, align them to a common daily grid on
    the high-resolution CHL reference grid, and gap-fill each variable with
    per-pixel spatio-temporal interpolation plus a domain-mean fallback.

    Parameters
    ----------
    bbox : list[float]
        [west, south, east, north] in degrees (WGS-84).
    time_range : tuple[str, str]
        Start and end dates, e.g. ('2021-05-01', '2021-09-01').
    resolution : float or None
        Output grid spacing in degrees.  ``None`` (default and recommended)
        uses the native s3_olci_chl grid — the most accurate, coastline-
        resolving choice.  A value (e.g. 0.003 ~= OLCI native) builds a regular
        grid at that spacing.
    dask_chunks : dict or None
        Dask chunk sizes for ``dc.load``.  Default ``{'time': -1}``.
    mask_land : bool
        If True, restore land / cloud-only pixels of SST and CHL to NaN so the
        coastline shows.  (Salinity is always left unmasked + extrapolated:
        its native ~2x3 grid is too coarse for a meaningful land mask.)
    verbose : bool
        If True, emit DEBUG-level progress.

    Returns
    -------
    xr.Dataset
        Variables: ``salinity``, ``temperature``, ``chlorophyll``,
        ``ocean_mask``.  Temperature in degC; land pixels NaN (white on maps).

    Notes
    -----
    DINEOF-era keyword arguments (``n_modes``, ``max_iterations``,
    ``tolerance``, ``cross_validation_fraction``, ``random_seed``) are accepted
    and ignored for backward compatibility.
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
        log.debug("Ignoring legacy DINEOF kwargs: %s", sorted(_legacy_ignored))

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

    # CHL is the reference grid (highest resolution -> resolves the coastline).
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

    log.info("Gridding + gap-filling each variable...")
    _t = time.time()

    # Salinity: native grid too coarse to mask -> extrapolate, leave unmasked.
    salinity_grid, _sal_valid = _gridded_filled(
        sss_data[SSS_VAR],
        ref_lats,
        ref_lons,
        time_horizon,
        mask_land=False,
        extrapolate=True,
    )
    # Temperature: Kelvin -> degC, land-masked.
    temperature_grid, temp_valid = _gridded_filled(
        sst_data[SST_VAR],
        ref_lats,
        ref_lons,
        time_horizon,
        kelvin_to_celsius=True,
        mask_land=mask_land,
    )
    # Chlorophyll: land-masked.
    chl_grid, chl_valid = _gridded_filled(
        chl_data[CHL_VAR],
        ref_lats,
        ref_lons,
        time_horizon,
        mask_land=mask_land,
    )

    # Sea mask from the high-resolution products (SST | CHL real observations).
    ocean_mask = (temp_valid | chl_valid).rename("ocean_mask")

    result = xr.Dataset(
        {
            "salinity": salinity_grid,
            "temperature": temperature_grid,
            "chlorophyll": chl_grid,
            "ocean_mask": ocean_mask,
        }
    ).compute()  # materialise the lazy dask arrays

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
        {v: int(result[v].isnull().sum()) for v in ("salinity", "temperature", "chlorophyll")},
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
        description="Load and gap-fill SSS/SST/CHL from the ODC (notebook method)."
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
        default=None,
        help="Grid spacing in degrees (default: native CHL grid)",
    )
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

    combined_data = load_filled_grid(
        bbox=args.bbox,
        time_range=(args.start, args.end),
        resolution=args.resolution,
        mask_land=not args.no_mask_land,
        verbose=args.verbose,
    )

    log.info("Result:\n%s", combined_data)

    if args.output:
        combined_data.to_netcdf(args.output)
        log.info("Saved to %s", args.output)
