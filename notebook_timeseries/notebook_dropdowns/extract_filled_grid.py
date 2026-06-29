"""
Extract and gap-fill a multi-variable ocean dataset from Open Data Cube.

Usage
-----
As a script:
    python extract_filled_grid.py \
        --bbox 10.706177 56.857983 12.683716 58.788132 \
        --start 2020-04-01 --end 2020-10-31 \
        [--resolution 0.125] [--seed 42] [--verbose]

As a module:
    from utils.extract_filled_grid import load_filled_grid
    ds = load_filled_grid(
        bbox=[10.706177, 56.857983, 12.683716, 58.788132],
        time_range=('2020-04-01', '2020-10-31'),
        verbose=True,
    )
"""

import argparse
import logging
import time

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

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
        n_valid / data_ocean.size * 100,
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
    final_cov = valid_ocean.sum() / (ocean_3d.sum() * nt) * 100
    log.info("    Final coverage (ocean only): %.2f%%", final_cov)
    return result


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def load_filled_grid(
    bbox,
    time_range,
    resolution=0.125,
    dask_chunks=None,
    n_modes="auto",
    max_iterations=100,
    tolerance=1e-4,
    cross_validation_fraction=0.05,
    random_seed=42,
    verbose=False,
):
    """
    Load SSS, SST, and CHL from ODC, align to a common daily grid, and
    gap-fill each variable using DINEOF.

    Parameters
    ----------
    bbox : list[float]
        [west, south, east, north] in degrees (WGS-84).
    time_range : tuple[str, str]
        Start and end dates, e.g. ('2020-04-01', '2020-10-31').
    resolution : float
        Output grid resolution in degrees (default 0.125).
    dask_chunks : dict or None
        Dask chunk sizes, e.g. {"latitude": 2048, "longitude": 2048}.
    n_modes : int or 'auto'
        Number of EOF modes for DINEOF. 'auto' selects via cross-validation.
    max_iterations : int
        Maximum DINEOF iterations per variable.
    tolerance : float
        Convergence tolerance for DINEOF.
    cross_validation_fraction : float
        Fraction of valid pixels withheld for cross-validation.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        If True, set the module logger to DEBUG level for detailed output.

    Returns
    -------
    xr.Dataset
        Gap-filled dataset with variables: salinity, temperature, chlorophyll,
        ocean_mask.  Land pixels remain NaN.
    """
    import warnings
    import datacube
    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    if verbose:
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            log.addHandler(logging.StreamHandler())
    # make datacube less pink
    logging.getLogger("datacube").setLevel(logging.ERROR)

    np.random.seed(random_seed)

    if dask_chunks is None:
        dask_chunks = {"latitude": 2048, "longitude": 2048}

    west, south, east, north = bbox
    dc = datacube.Datacube()

    load_kwargs = dict(
        x=(west, east),
        y=(south, north),
        time=time_range,
        resolution=resolution,
        output_crs="EPSG:4326",
        dask_chunks=dask_chunks,
    )

    _t_load = time.time()

    log.info("Loading SSS (cmems_sss)...")
    _t = time.time()
    sss_data = dc.load(product="cmems_sss", measurements=["sos"], **load_kwargs)
    log.debug("  SSS loaded in %.1fs", time.time() - _t)

    log.info("Loading SST (s3_slstr_sst)...")
    _t = time.time()
    sst_data = dc.load(
        product="s3_slstr_sst", measurements=["sea_surface_temperature"], **load_kwargs
    )
    log.debug("  SST loaded in %.1fs", time.time() - _t)

    log.info("Loading CHL (cmems_chl_tur)...")
    _t = time.time()
    chl_data = dc.load(product="cmems_chl_tur", measurements=["CHL"], **load_kwargs)
    log.debug("  CHL loaded in %.1fs", time.time() - _t)

    # Resample swath products to daily mean
    sst_data = sst_data.resample(time="1D").mean()
    chl_data = chl_data.resample(time="1D").mean()

    # K -> °C
    sst_data["sea_surface_temperature"] = sst_data["sea_surface_temperature"] - 273.15
    sst_data["sea_surface_temperature"].attrs["units"] = "degrees Celsius"

    # Floor all timestamps to day
    sss_data["time"] = sss_data["time"].dt.floor("D")
    sst_data["time"] = sst_data["time"].dt.floor("D")
    chl_data["time"] = chl_data["time"].dt.floor("D")

    # Align to common time axis
    all_times = np.sort(
        np.unique(
            np.concatenate(
                [
                    sss_data.time.values,
                    sst_data.time.values,
                    chl_data.time.values,
                ]
            )
        )
    )
    sss_data = sss_data.reindex(time=all_times, fill_value=np.nan)
    sst_data = sst_data.reindex(time=all_times, fill_value=np.nan)
    chl_data = chl_data.reindex(time=all_times, fill_value=np.nan)

    log.info(
        "Aligned to %d daily time steps (%s – %s)",
        len(all_times),
        str(all_times[0])[:10],
        str(all_times[-1])[:10],
    )

    combined_data = xr.Dataset(
        {
            "salinity": sss_data["sos"].rename("salinity"),
            "temperature": sst_data["sea_surface_temperature"].rename("temperature"),
            "chlorophyll": chl_data["CHL"].rename("chlorophyll"),
        }
    )

    # Ocean mask: pixels with at least one valid observation across any variable/time
    ocean_mask_da = np.isfinite(combined_data.to_array(dim="variable")).any(
        ("variable", "time")
    )
    ocean_mask_da.name = "ocean_mask"
    ocean_mask_da = ocean_mask_da.compute()  # compute once; reused below and in DINEOF
    n_ocean = int(ocean_mask_da.values.sum())
    n_total = ocean_mask_da.size
    log.info(
        "Ocean mask: %d/%d pixels (%.1f%% ocean)",
        n_ocean,
        n_total,
        n_ocean / n_total * 100,
    )

    # Check each variable has at least some valid data
    empty = [
        v
        for v in ("salinity", "temperature", "chlorophyll")
        if not np.isfinite(combined_data[v]).any()
    ]
    if empty:
        raise ValueError(
            f"No data found for variable(s): {', '.join(empty)}. "
            "Check that the bbox and time_range overlap indexed products."
        )

    # Compute dask arrays before DINEOF (DINEOF works on numpy)
    log.debug("Computing data arrays...")
    _t = time.time()
    combined_data = combined_data.compute()
    log.debug("  Compute done in %.1fs", time.time() - _t)

    # DINEOF gap-filling
    log.info("Starting DINEOF gap-filling...")
    t0 = time.time()

    filled = {}
    _dineof_times = {}
    for var_name in ("salinity", "temperature", "chlorophyll"):
        _t = time.time()
        filled[var_name] = dineof_reconstruction(
            combined_data[var_name],
            ocean_mask_da,
            n_modes=n_modes,
            max_iterations=max_iterations,
            tolerance=tolerance,
            cross_validation_fraction=cross_validation_fraction,
        )
        _dineof_times[var_name] = time.time() - _t
        log.info("  %s done in %.1fs", var_name.capitalize(), _dineof_times[var_name])

    _t_total = time.time() - t0
    log.info("Total DINEOF time: %.2f min", _t_total / 60)

    if log.isEnabledFor(logging.DEBUG):
        _t_load_total = t0 - _t_load  # time from first load to DINEOF start
        log.debug(
            "\n  Timing summary"
            "\n  %-13s %6.1fs  (%4.2f min)"
            "\n  %-13s %6.1fs  (%4.2f min)"
            "\n  %-13s %6.1fs  (%4.2f min)"
            "\n  %-13s %6.1fs  (%4.2f min)"
            "\n  %-13s %6.1fs  (%4.2f min)",
            "Load+prep:",
            _t_load_total,
            _t_load_total / 60,
            "Salinity:",
            _dineof_times["salinity"],
            _dineof_times["salinity"] / 60,
            "Temperature:",
            _dineof_times["temperature"],
            _dineof_times["temperature"] / 60,
            "Chlorophyll:",
            _dineof_times["chlorophyll"],
            _dineof_times["chlorophyll"] / 60,
            "TOTAL:",
            _t_load_total + _t_total,
            (_t_load_total + _t_total) / 60,
        )

    result = xr.Dataset(
        {
            "salinity": filled["salinity"],
            "temperature": filled["temperature"],
            "chlorophyll": filled["chlorophyll"],
            "ocean_mask": ocean_mask_da,
        }
    )

    # Fallback: fill any residual ocean NaNs with temporal mean
    ocean_2d = ocean_mask_da.values
    for var_name in ("salinity", "temperature", "chlorophyll"):
        ocean_3d = np.broadcast_to(ocean_2d[np.newaxis, :, :], result[var_name].shape)
        residual = np.isnan(result[var_name].values) & ocean_3d
        if residual.any():
            log.warning(
                "Applying fallback fill for %s (%d pixels)...",
                var_name,
                residual.sum(),
            )
            tmean = np.nanmean(result[var_name].values, axis=0)
            vals = result[var_name].values.copy()
            mask = np.isnan(vals) & ocean_3d
            vals[mask] = np.broadcast_to(tmean, vals.shape)[mask]
            global_ocean_mean = np.nanmean(vals[ocean_3d])
            vals[np.isnan(vals) & ocean_3d] = global_ocean_mean
            result[var_name] = xr.DataArray(
                vals,
                coords=result[var_name].coords,
                dims=result[var_name].dims,
                attrs=result[var_name].attrs,
            )

    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_timestep(dataset, selected_date, return_fig=True):
    """
    Plot salinity, temperature, and chlorophyll from *dataset* at a single date.

    Parameters
    ----------
    dataset : xr.Dataset
        Output of :func:`load_filled_grid` — must contain variables
        ``salinity``, ``temperature``, and ``chlorophyll``.
    selected_date : str, datetime.date, datetime.datetime, or numpy.datetime64
        Date to plot.  The nearest available time step is selected.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    selected_dt = np.datetime64(str(selected_date), "ns")
    time_index = int(np.argmin(np.abs(dataset.time.values - selected_dt)))
    date_label = str(dataset.time.values[time_index])[:10]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    dataset["salinity"].isel(time=time_index).plot(
        ax=axes[0],
        cmap="viridis",
        vmin=18,
        vmax=32,
        cbar_kwargs={"label": "Salinity (PSU)"},
        transform=ccrs.PlateCarree(),
    )
    axes[0].coastlines(resolution="10m", linewidth=0.8, color="black")
    axes[0].add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
    axes[0].set_title(f"Sea Surface Salinity\n{date_label}")

    dataset["temperature"].isel(time=time_index).plot(
        ax=axes[1],
        cmap="RdYlBu_r",
        vmin=10,
        vmax=24,
        cbar_kwargs={"label": "Temperature (°C)"},
        transform=ccrs.PlateCarree(),
    )
    axes[1].coastlines(resolution="10m", linewidth=0.8, color="black")
    axes[1].add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
    axes[1].set_title(f"Sea Surface Temperature\n{date_label}")

    dataset["chlorophyll"].isel(time=time_index).plot(
        ax=axes[2],
        cmap="YlGn",
        norm=plt.matplotlib.colors.LogNorm(vmin=0.01, vmax=67.0),
        cbar_kwargs={"label": "Chlorophyll (mg/m³)"},
        transform=ccrs.PlateCarree(),
    )
    axes[2].coastlines(resolution="10m", linewidth=0.8, color="black")
    axes[2].add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
    axes[2].set_title(f"Chlorophyll (log scale)\n{date_label}")

    plt.tight_layout()
    plt.show()
    if return_fig:
        return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Load and DINEOF gap-fill SSS/SST/CHL from ODC."
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
        help="Bounding box in WGS-84 degrees",
    )
    p.add_argument(
        "--start", required=True, metavar="DATE", help="Start date (YYYY-MM-DD)"
    )
    p.add_argument("--end", required=True, metavar="DATE", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--resolution",
        type=float,
        default=0.125,
        help="Grid resolution in degrees (default: 0.125)",
    )
    p.add_argument(
        "--modes", default="auto", help="Number of EOF modes, or 'auto' (default: auto)"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Optional NetCDF output path (e.g. combined_data.nc)",
    )
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
        verbose=args.verbose,
    )

    log.info("Result:\n%s", combined_data)

    if args.output:
        combined_data.to_netcdf(args.output)
        log.info("Saved to %s", args.output)
