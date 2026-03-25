import numpy as np
import pandas as pd
import time
from punk.period import get_multiterm_period_estimate, perform_residual_resampling
import punk.utils as utils
from phunk.reparametrization import compute_LU_bounds
from phunk.geometry import estimate_axes_ratio


def initialize(
    phase_curve, remap=True, weights=None, metadata=False, force_period=None
):
    """
    Initialize a SOCCA model fit using SHG1G2 pre-fitting, period estimation,
    and pole/period grid search.

    This function performs a preliminary sHG1G2 fit to the input phase curve,
    computes residuals, and derives an initial estimate of the synodic period.
    It then explores a grid of trial rotation periods and spin pole orientations
    to determine optimal starting parameters for a SOCCA inversion.

    The initialization includes:
    - Residual-based period estimation (or user-imposed period)
    - Determination of a period scan range based on heliocentric distance
    - Construction of an RMS landscape over spin pole coordinates
    - Detection of local minima in pole space for initialization
    - Estimation of shape parameters (axis ratios)
    - Grid search over (period, pole) combinations to minimize SOCCA RMS

    Parameters
    ----------
    phase_curve : object
        Phase curve object containing photometric and geometric data.
        Must provide attributes and methods:
        - `phase`, `ra`, `dec`, `mag`, `mag_err`, `epoch`
        - `band`, `bands`
        - `Dhelio`
        - `fit(models=..., p0=..., remap=..., weights=...)`
        - fitted model access via `phase_curve.sHG1G2` and `phase_curve.SOCCA`
    remap : bool, optional
        Whether to remap parameters during SOCCA fitting. Default is True.
    weights : array-like or None, optional
        Optional weights for the SOCCA fit. Default is None.
    metadata : bool, optional
        If True, return diagnostic information including execution time and
        bootstrap score from period estimation. Default is False.
    force_period : float or None, optional
        If provided, bypass automatic period estimation and use this value
        (in days) as the central period for the scan. Default is None.

    Returns
    -------
    opt_p0 : dict
        Dictionary of optimal initial parameters for the SOCCA model, including:
        - Photometric parameters (H, G1, G2 per band)
        - Spin parameters (`period`, `alpha`, `delta`)
        - Shape parameters (`a_b`, `a_c`)
        - Initial rotation phase (`W0`)
    QA_dict : dict
        Dictionary containing quality assessment metrics. Returned empty if
        `metadata=False`. Otherwise may include:
        - "Inversion time (seconds)"
        - "Bootstrap score"
    """
    # We give the phase curve our sHG1G2 attributes
    if metadata:
        t1 = time.time()

    phase_curve.fit(models=["sHG1G2"])

    bands = np.asarray(phase_curve.band)
    residuals = np.zeros(len(bands))

    for band in phase_curve.bands:
        mask = bands == band

        model = phase_curve.sHG1G2.eval(
            phase_curve.phase[mask],
            phase_curve.ra[mask],
            phase_curve.dec[mask],
            band=band,
        )

        residuals[mask] = model - phase_curve.mag[mask]

    residuals_dataframe = pd.DataFrame(
        {
            "jd": phase_curve.epoch,
            "residuals": residuals,
            "filters": bands,
            "sigma": phase_curve.mag_err,
        }
    )
    # Period search boundaries (in days)
    if force_period is None:
        pmin, pmax = 5e-2, 1e4
        try:
            p_in, k_val, p_rms, signal_peaks, window_peaks = (
                get_multiterm_period_estimate(
                    residuals_dataframe, p_min=pmin, p_max=pmax, k_free=True
                )
            )
            if metadata:
                _, Nbs = perform_residual_resampling(
                    resid_df=residuals_dataframe,
                    p_min=pmin,
                    p_max=pmax,
                    k=int(k_val),
                )
        except KeyError:
            # If more than 10 terms are required switch to fast rotator:
            pmin, pmax = 5e-3, 5e-2

            p_in, k_val, p_rms, signal_peaks, window_peaks = (
                get_multiterm_period_estimate(
                    residuals_dataframe, p_min=pmin, p_max=pmax, k_free=True
                )
            )
            if metadata:
                _, Nbs = perform_residual_resampling(
                    resid_df=residuals_dataframe,
                    p_min=pmin,
                    p_max=pmax,
                    k=int(k_val),
                )
    else:
        p_in = force_period
    period_sy = p_in
    # Add heliocentric distance mean
    sma = phase_curve.Dhelio.mean()  # in AU

    W = utils.period_range(sma, period_sy * 24) / 24  # in days
    N = utils.Nintervals(sma)

    Pmin = period_sy - W
    Pmax = period_sy + W

    period_scan = np.linspace(Pmin, Pmax, N)

    if not np.isclose(period_scan, period_sy).any():
        period_scan = np.sort(np.append(period_scan, period_sy))

    ra0, dec0 = phase_curve.sHG1G2.alpha, phase_curve.sHG1G2.delta

    rarange = np.arange(0, 360, 10)
    decrange = np.arange(-90, 90, 5)
    rms_landscape = np.ones(shape=(len(rarange), len(decrange)))

    # Initialize axes ratios from lightcurve amplitude
    for band in phase_curve.bands:
        mask = bands == band

        model = phase_curve.sHG1G2.eval(
            phase_curve.phase[mask],
            phase_curve.ra[mask],
            phase_curve.dec[mask],
            band=band,
        )

        residuals[mask] = model - phase_curve.mag[mask]

    a_b, a_c = estimate_axes_ratio(residuals, phase_curve.sHG1G2.R)

    for i, ra0 in enumerate(rarange):
        for j, dec0 in enumerate(decrange):
            all_residuals = []

            for band in phase_curve.bands:
                mask = bands == band

                model = phase_curve.sHG1G2.eval(
                    phase_curve.phase[mask],
                    phase_curve.ra[mask],
                    phase_curve.dec[mask],
                    band=band,
                    alpha=ra0,
                    delta=dec0,
                )

                obs = phase_curve.mag[mask]

                all_residuals.append(obs - model)

            all_residuals = np.concatenate(all_residuals)
            rms_landscape[j, i] = np.sqrt(np.mean(all_residuals**2))

    interp_vals = utils.gaussian_interpolate(rms_landscape, factor=4, sigma=1.0)
    ny, nx = interp_vals.shape
    ra_vals = np.linspace(rarange.min(), rarange.max(), nx)
    dec_vals = np.linspace(decrange.min(), decrange.max(), ny)
    ys, xs = utils.detect_local_minima(interp_vals)
    ra_minima = ra_vals[xs]
    dec_minima = dec_vals[ys]

    ra_init = ra_minima
    dec_init = dec_minima

    # Add near-pole initialization points
    ra_init = np.append(ra_init, 220)
    ra_init = np.append(ra_init, 140)

    dec_init = np.append(dec_init, 70)
    dec_init = np.append(dec_init, -70)

    # Remove pairs at the parameter space border
    RA_MARGIN = 1.0  # degrees

    ra_mask = (ra_init > RA_MARGIN) & (ra_init < 360 - RA_MARGIN)

    ra_init = ra_init[ra_mask]
    dec_init = dec_init[ra_mask]

    H_vals = [getattr(phase_curve.sHG1G2, f"H{band}") for band in phase_curve.bands]
    G1_vals = [getattr(phase_curve.sHG1G2, f"G1{band}") for band in phase_curve.bands]
    G2_vals = [getattr(phase_curve.sHG1G2, f"G2{band}") for band in phase_curve.bands]

    for i, (G1, G2) in enumerate(zip(G1_vals, G2_vals)):
        L, U = compute_LU_bounds(G1)
        tol = 5e-2
        GMIN = -0.429
        GMAX = 1.429

        if G1 < GMIN + tol or G1 > GMAX - tol or G2 < L + tol or G2 > U - tol:
            G1 = 0.15
            L, U = compute_LU_bounds(G1)
            G2 = (L + U) / 2

            G1_vals[i] = G1
            G2_vals[i] = G2

    bands = np.asarray(phase_curve.band)
    residuals = np.zeros(len(bands))

    for band in phase_curve.bands:
        mask = bands == band

        model = phase_curve.sHG1G2.eval(
            phase_curve.phase[mask],
            phase_curve.ra[mask],
            phase_curve.dec[mask],
            band=band,
        )

        residuals[mask] = model - phase_curve.mag[mask]

    a_b, a_c = estimate_axes_ratio(residuals, phase_curve.sHG1G2.R)

    if (not (1 <= a_b <= 5 and 1 <= a_c <= 5)) or np.isclose(
        a_b, a_c, rtol=1e-6, atol=1e-9
    ):
        a_b = 1.05
        a_c = 1.5

    opt_rms = np.inf
    opt_p0 = None

    for ra, dec in zip(ra_init, dec_init):
        for period_sc in period_scan:
            p_in = {}

            for band, H in zip(phase_curve.bands, H_vals):
                p_in[f"H{band}"] = H

            for band, G1 in zip(phase_curve.bands, G1_vals):
                p_in[f"G1{band}"] = G1

            for band, G2 in zip(phase_curve.bands, G2_vals):
                p_in[f"G2{band}"] = G2

            p_in.update(
                {
                    "period": period_sc,
                    "alpha": ra,
                    "delta": dec,
                    "a_b": a_b,
                    "a_c": a_c,
                    "W0": np.rad2deg(0.1),
                }
            )
            phase_curve.fit(models=["SOCCA"], p0=p_in, remap=remap, weights=weights)

            current_rms = phase_curve.SOCCA.rms
            if current_rms < opt_rms:
                opt_rms = current_rms
                opt_p0 = p_in

    QA_dict = {}
    if metadata:
        t2 = time.time()
        QA_dict = {"Inversion time (seconds)": t2 - t1, "Bootstrap score": Nbs}

    return opt_p0, QA_dict
