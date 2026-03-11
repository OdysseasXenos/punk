import numpy as np
import pandas as pd
import time
from period import get_multiterm_period_estimate, perform_residual_resampling
import utils
from phunk.reparametrization import compute_LU_bounds
from phunk.geometry import estimate_axes_ratio


def initialize(phase_curve, remap=True, weights=None, metadata=False):
    """
    Fit a small solar system object's photometric data using SHG1G2 or SOCCA models.

    This function can perform either a standard SHG1G2 fit or a spin- and
    shape-constrained SOCCA fit, optionally including blind scans over
    initial pole positions and periods. It supports filtering data by survey.

    Parameters
    ----------
    data : pandas.DataFrame single-row
        Input dataset containing photometry and geometry with columns:
        - 'cmred': reduced magnitudes
        - 'csigmapsf': uncertainties
        - 'Phase': solar phase angles (deg)
        - 'cfid': filter IDs
        - 'ra', 'dec': coordinates (deg)
        - 'cjd': observation times (light-time corrected)
        Optional (for terminator fits):
        - 'ra_s', 'dec_s': sub-solar point coordinates (deg)
    flavor : str
        Model type to fit. Must be 'SHG1G2' or 'SOCCA'.
    shg1g2_constrained : bool, optional
        Whether to constrain the SOCCA fit using a prior SHG1G2 solution. Default True.
    period_blind : bool, optional
        If True, perform a small grid search over initial periods. Default True.
    pole_blind : bool, optional
        If True, perform a grid search over 12 initial poles all over a sphere. Default True.
        If False, produce the sHG1G2 rms error landscape and initialize SOCCA poles on its local minima
    p0 : list, optional
        Initial guess parameters for the fit. Required if `shg1g2_constrained=False`.
    alt_spin : bool, optional
        For SOCCA constrained fits, use the antipodal spin solution. Default False.
    period_in : float, optional
        Input synodic period (days) to override automatic estimation. Default None.
    period_quality_flag : bool, optional
        Provide bootstrap score, alias/true (0/1) flags and period fit rms for the period estimates
    terminator : bool, optional
        If True, include self-shading in the fit. Default False.
    time_me : bool, optional
        If True, include timing (in seconds). Default True.

    Returns
    -------
    dict or tuple
        If `flavor='SHG1G2'`:
            dict
                Best-fit SHG1G2 parameters.
        If `flavor='SOCCA'`:
            dict
                Best-fit SOCCA parameters.

    Notes
    -----
    - For SOCCA fits with `shg1g2_constrained=True`, the function first performs
      a SHG1G2 fit to constrain H, G1, G2, and shape parameters.
    - Blind scans systematically vary initial pole positions and period to find
      the optimal fit when `blind_scan=True`.

    Raises
    ------
    ValueError
        If `flavor` is not 'SHG1G2' or 'SOCCA'.
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
    pmin, pmax = 5e-2, 1e4
    try:
        p_in, k_val, p_rms, signal_peaks, window_peaks = get_multiterm_period_estimate(
            residuals_dataframe, p_min=pmin, p_max=pmax, k_free=True
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

        p_in, k_val, p_rms, signal_peaks, window_peaks = get_multiterm_period_estimate(
            residuals_dataframe, p_min=pmin, p_max=pmax, k_free=True
        )
        if metadata:
            _, Nbs = perform_residual_resampling(
                resid_df=residuals_dataframe,
                p_min=pmin,
                p_max=pmax,
                k=int(k_val),
            )
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

    ############################
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
    ############################

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
