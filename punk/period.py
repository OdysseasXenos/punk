import numpy as np
import pandas as pd

from astropy.timeseries import LombScargleMultiband, LombScargle
from scipy.signal import find_peaks
import nifty_ls  # noqa: F401
from scipy.stats import f as ftest

import logging


def alias_func(x, i, j, p_feat):
    """
    Compute the aliasing relation for a periodic signal.

    Parameters
    ----------
    x : float
        Input period.
    i : int
        Positive integer representing the mode number.
    j : int
        Alias index.
    p_feat : float
        Characteristic frequency of the feature (e.g. 1 day).

    Returns
    -------
    float
        Value of the aliasing relation for the given inputs.
    """
    return (i + 1) * p_feat * x / np.abs(p_feat - j * x)


def get_period_estimate(residuals_dataframe, p_min=0.03, p_max=2):
    """
    Estimate significant periods in a time series of residuals using Lomb-Scargle periodograms.

    Parameters
    ----------
    residuals_dataframe : pandas.DataFrame
        DataFrame containing at least the following columns:
        - 'jd' : observation times
        - 'residuals' : residuals (observed - SHG1G2 modeled magnitudes)
        - 'filters' : filter IDs (int)
    p_min : float, optional
        Minimum period to search (in days). Default is 0.03.
    p_max : float, optional
        Maximum period to search (in days). Default is 2.

    Returns
    -------
    tuple
        - single_band_results : list
            [frequencies, power, top signal peak frequencies, corresponding powers]
        - window_function_results : list
            [frequencies, power, top window peak frequencies, corresponding powers]
        - noise_level : float
            Estimated noise level in the periodogram (mean + 3*std of power)

    Notes
    -----
    - Uses `LombScargle` for the window function and `LombScargleMultiband` from the `nifty-ls` implementations
    - The five strongest peaks are returned.
    """
    period_min, period_max = p_min, p_max
    period_range = (period_min, period_max)

    ls = LombScargle(
        residuals_dataframe["jd"].values,
        np.ones(len(residuals_dataframe["jd"])),
        fit_mean=False,
        center_data=False,
    )
    frequencyW, powerW = ls.autopower(
        minimum_frequency=1 / period_range[1],
        maximum_frequency=1 / period_range[0],
        method="fastnifty",
    )
    model = LombScargleMultiband(
        residuals_dataframe["jd"].values,
        residuals_dataframe["residuals"].values,
        residuals_dataframe["filters"].values,
        residuals_dataframe["sigma"].values,
        normalization="psd",
        fit_mean=True,
        nterms_base=1,
        nterms_band=1,
    )
    frequency, power = model.autopower(
        method="fast",
        sb_method="fastnifty",
        minimum_frequency=1 / period_range[1],
        maximum_frequency=1 / period_range[0],
        samples_per_peak=5,
    )
    pindex, heights = find_peaks(
        power,
        height=0.1,
        threshold=None,
        distance=260,
    )
    pindexW, heightsW = find_peaks(
        powerW,
        height=0.1,
        threshold=None,
        distance=260,
    )
    hindex = np.argsort(heights["peak_heights"])[::-1][:5]
    hindexW = np.argsort(heightsW["peak_heights"])[::-1][:5]

    signal_peaks = frequency[pindex[hindex]]
    signal_power = power[pindex[hindex]]

    window_peaks = frequencyW[pindexW[hindexW]]
    window_power = powerW[pindexW[hindexW]]

    noise_level = np.mean(power) + 3 * np.std(power)

    return (
        [frequency, power, signal_peaks, signal_power],
        [frequencyW, powerW, window_peaks, window_power],
        noise_level,
    )


def get_multiterm_period_estimate(
    residuals_dataframe, p_min=0.03, p_max=2, k_free=True, k_val=None, k_max=4
):
    """
    Estimate the period of a multiband time series using a multiband Lomb-Scargle model.

    Fits a multiband Lomb-Scargle model to the SHG1G2 residuals across different filters,
    optionally testing multiple base-term complexities (`k`) and selecting the simplest model
    that adequately describes the data.

    Parameters
    ----------
    residuals_dataframe : pandas.DataFrame
        DataFrame containing:
        - 'jd' : observation times
        - 'residuals' : residual magnitudes (observed - model)
        - 'filters' : filter IDs
        - 'sigma' : observational uncertainties
    p_min : float, optional
        Minimum period to search (days). Default is 0.03.
    p_max : float, optional
        Maximum period to search (days). Default is 2.
    k_free : bool, optional
        If True, automatically scan multiple base-term complexities to choose optimal model. Default True.
    k_val : int, optional
        Fixed number of base terms to use if `k_free=False`. Default None.
    k_max : int
        Maximum number of terms to use for the multiterm LS periodogram. Default 4
    Returns
    -------
    tuple
        - period_in : float
            Estimated dominant SSO period (2 / f_best) in the time series.
        - k_val : int
            Number of base terms used in the final multiband model.
        - p_rms : float
            RMS of residuals for the chosen model (NaN if k_free=False).
        - signal_peaks : array_like
            Frequencies of the top five peaks in the final multiband periodogram.
        - window_peaks : array_like
            Frequencies of the top five peaks in the single-band window function periodogram.

    Notes
    -----
    - Uses `LombScargle` for the window function and `LombScargleMultiband` from the `nifty-ls` implementations
    - If `k_free=True`, performs F-test comparisons to select the simplest adequate model.
    """

    period_min, period_max = p_min, p_max
    period_range = (period_min, period_max)
    results = []
    residuals = np.zeros(len(residuals_dataframe["filters"].values))
    bands = np.unique(residuals_dataframe["filters"].values)
    if k_free:
        for k in range(1, k_max + 1):
            model = LombScargleMultiband(
                residuals_dataframe["jd"].values,
                residuals_dataframe["residuals"].values,
                residuals_dataframe["filters"].values,
                residuals_dataframe["sigma"].values,
                normalization="standard",
                fit_mean=True,
                nterms_base=k,
                nterms_band=1,
            )
            try:
                frequency, power = model.autopower(
                    method="fast",
                    sb_method="fastnifty_chi2",
                    minimum_frequency=1 / period_range[1],
                    maximum_frequency=1 / period_range[0],
                    samples_per_peak=5,
                )
            except np.linalg.LinAlgError:
                logging.warning("Singular matrix for k={}".format(k))
                continue

            f_best = frequency[np.argmax(power)]
            y_model = model.model(
                residuals_dataframe["jd"].values, f_best, bands_fit=bands
            )

            # y_model.shape is (n_bands, len(time))

            for n, ff in enumerate(bands):
                bindex = np.where(residuals_dataframe["filters"].values == ff)
                residuals[bindex] = (
                    residuals_dataframe["residuals"].values[bindex] - y_model[n][bindex]
                )

            rms = np.sqrt(np.mean(residuals**2))
            n_params = 2 * k + 1 + 3 * len(bands)
            dof = n_params
            N = len(residuals)

            results.append((k, f_best, rms, dof, n_params))

        model_comparison = pd.DataFrame()

        for i in range(len(results) - 1):
            k, f_best, rss, dof, n_params = results[i]
            k_next, f_best_next, rss_next, dof_next, n_params_next = results[i + 1]
            F = ((rss**2 / rss_next**2) - 1) * ((N - dof_next) / (dof_next - dof))

            # Here crit = Fstat value for which model_2 (more complex) is in fact better than model_1 (less complex)
            crit = ftest.ppf(
                q=0.99,
                dfn=n_params_next - n_params,
                dfd=N - n_params_next,
            )

            model_comparison.loc[i, "k"] = k
            model_comparison.loc[i, "k_next"] = k_next
            model_comparison.loc[i, "f_best"] = f_best
            model_comparison.loc[i, "Fstat"] = F
            model_comparison.loc[i, "alpha_crit"] = crit
            model_comparison.loc[i, "rms"] = rss

        cond = model_comparison["Fstat"] > model_comparison["alpha_crit"]

        model_comparison = model_comparison[
            ~cond
        ]  # don't go for the more complex model
        f_chosen = model_comparison.loc[model_comparison.k.idxmin()]["f_best"]
        k_val = model_comparison.loc[model_comparison.k.idxmin()]["k"]
        p_rms = model_comparison.loc[model_comparison.k.idxmin()]["rms"]
    if not k_free:
        model = LombScargleMultiband(
            residuals_dataframe["jd"].values,
            residuals_dataframe["residuals"].values,
            residuals_dataframe["filters"].values,
            residuals_dataframe["sigma"].values,
            normalization="standard",
            fit_mean=True,
            nterms_base=int(k_val),
            nterms_band=1,
        )
        frequency, power = model.autopower(
            method="fast",
            sb_method="fastnifty_chi2",
            minimum_frequency=1 / period_range[1],
            maximum_frequency=1 / period_range[0],
            samples_per_peak=5,
        )

        f_best = frequency[np.argmax(power)]
        f_chosen = f_best
        p_rms = np.nan

    period_in = 2 * (1 / f_chosen)

    model_final = LombScargleMultiband(
        residuals_dataframe["jd"].values,
        residuals_dataframe["residuals"].values,
        residuals_dataframe["filters"].values,
        residuals_dataframe["sigma"].values,
        normalization="standard",
        fit_mean=True,
        nterms_base=int(k_val),
        nterms_band=1,
    )
    frequency_final, power_final = model_final.autopower(
        method="fast",
        sb_method="fastnifty_chi2",
        minimum_frequency=1 / period_range[1],
        maximum_frequency=1 / period_range[0],
        samples_per_peak=5,
    )

    ls = LombScargle(
        residuals_dataframe["jd"].values,
        np.ones(len(residuals_dataframe["jd"])),
        fit_mean=False,
        center_data=False,
    )
    frequencyW, powerW = ls.autopower(
        minimum_frequency=1 / period_range[1],
        maximum_frequency=1 / period_range[0],
        method="fastnifty",
    )

    pindex, heights = find_peaks(
        power_final,
        height=0.1,
        threshold=None,
        distance=260,
    )
    pindexW, heightsW = find_peaks(
        powerW,
        height=0.1,
        threshold=None,
        distance=260,
    )
    hindex = np.argsort(heights["peak_heights"])[::-1][:5]
    hindexW = np.argsort(heightsW["peak_heights"])[::-1][:5]

    signal_peaks = frequency_final[pindex[hindex]]
    window_peaks = frequencyW[pindexW[hindexW]]

    return period_in, int(k_val), p_rms, signal_peaks, window_peaks


def perform_residual_resampling(resid_df, p_min, p_max, k=1):
    """
    Estimate the robustness of a period measurement via bootstrap resampling of residuals.

    This function resamples the residuals DataFrame multiple times with replacement,
    recomputes the period for each bootstrap sample, and counts how many of the
    resampled periods are within 1% of the original period. It supports both single-band
    (k=1) and multiband (k>1) period estimation.

    Parameters
    ----------
    resid_df : pandas.DataFrame
        DataFrame of residuals containing columns required for `get_period_estimate`
        or `get_multiterm_period_estimate`.
    p_min : float
        Minimum period to search (days).
    p_max : float
        Maximum period to search (days).
    k : int, optional
        Number of base terms in the model; k=1 for single-band, k>1 for multiband. Default is 1.

    Returns
    -------
    tuple
        - BS_df : pandas.DataFrame
            The last bootstrap-resampled residuals DataFrame.
        - Nbs : int
            Number of bootstrap samples whose estimated period is within 1% of the original period.

    Notes
    -----
    - Performs 25 bootstrap resamples by default.
    - For k=1, uses `get_period_estimate`; for k>1, uses `get_multiterm_period_estimate`.
    """

    if k == 1:
        sg, w, _ = get_period_estimate(resid_df)
        Pog = 48 / sg[2][0]  # in hours

        # Bootstrap residuals:
        Pbs = np.zeros(25)
        for n in range(25):
            BS_df = resid_df.sample(n=len(resid_df), replace=True)
            sg, w, _ = get_period_estimate(BS_df)
            Pbs[n] = 48 / sg[2][0]

        cond = np.abs(Pog - Pbs) / Pog < 1e-2
        Nbs = np.sum(np.ones(25)[cond])
    if k > 1:
        Ptmp, _, _, _, _ = get_multiterm_period_estimate(
            resid_df, p_min=p_min, p_max=p_max, k_free=False, k_val=k
        )
        Pog = 24 / Ptmp
        Pbs = np.zeros(25)
        for n in range(25):
            try:
                BS_df = resid_df.sample(n=len(resid_df), replace=True)
                Ptmp, _, _, _, _ = get_multiterm_period_estimate(
                    BS_df, p_min=p_min, p_max=p_max, k_free=False, k_val=k
                )
                Pbs[n] = 24 / Ptmp
            except Exception:
                Ptmp = 2 * Pog
                Pbs[n] = 24 / Ptmp
        cond = np.abs(Pog - Pbs) / Pog < 1e-2
        Nbs = np.sum(np.ones(25)[cond])
    return BS_df, Nbs
