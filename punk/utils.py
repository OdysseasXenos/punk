import numpy as np
import astropy.constants as const


def calculate_reduced_magnitude(magnitude, D_observer, D_sun):
    """
    Calculate reduced magnitude

    Parameters
    -----------------------------
    magnitude : float/np.array
        Mangitude from obervation
    D_oberver : float/np.array
        Object-observer distance
    D_sun : float/np.array
        Object-sun distance

    Returns
    -----------------------------
    float/np.array, reduced magnitude
    """
    return magnitude - 5 * np.log10(D_observer * D_sun)


def flip_spin(ra0, dec0):
    """
    Compute the antipodal (spin-flip) direction for a given RA/Dec.

    Parameters
    ----------
    ra0 : float or array_like
        Right ascension in degrees.
    dec0 : float or array_like
        Declination in degrees.

    Returns
    -------
    tuple of float or array_like
        `(ra_alt, dec_alt)` giving the antipodal right ascension and
        declination in degrees.
    """
    ra0 = np.radians(ra0)
    dec0 = np.radians(dec0)
    ra_alt = (ra0 + np.pi) % (2 * np.pi)
    dec_alt = -dec0
    return np.rad2deg(ra_alt), np.rad2deg(dec_alt)


def calc_atan_parameter(ra, dec, ra0, dec0):
    """
    Compute the angular parameter used in great-circle or pole-separation geometry.

    Parameters
    ----------
    ra : float or array_like
        Right ascension of the target point, in degrees.
    dec : float or array_like
        Declination of the target point, in degrees.
    ra0 : float or array_like
        Reference right ascension, in degrees.
    dec0 : float or array_like
        Reference declination, in degrees.

    Returns
    -------
    float or array_like
        The atan2-based angle (in radians) describing the orientation of
        the target point relative to the reference direction.
    """
    ra, dec, ra0, dec0 = (
        np.radians(ra),
        np.radians(dec),
        np.radians(ra0),
        np.radians(dec0),
    )
    x = np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    y = np.cos(dec) * np.sin(ra - ra0)
    return np.arctan2(x, y)


def angle_after_one_synodic_period(angle, synodic_period, rate):
    """
    Propagate an angle forward by one synodic period given a drift rate.

    Parameters
    ----------
    angle : float or array_like
        Initial angle in degrees.
    synodic_period : float
        Synodic period in days.
    rate : float or array_like
        Angular drift rate in arcseconds per minute.

    Returns
    -------
    float or array_like
        Angle after one synodic period, in degrees.
    """
    angle_t1 = (
        angle + synodic_period * (60 * 24) / 3600 * rate
    )  # dRA in arcsec/min, period in days, ra_t0|1 in degrees
    return angle_t1


def estimate_sidereal_period(data, model_parameters, synodic_period):
    """
    Estimate the sidereal period of an object from its astrometric drift
    over one synodic period, using both the nominal and spin-flipped
    pole solutions.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing at least the columns:
        - 'cjd' : float
            Corrected Julian dates.
        - 'ra' : float
            Right ascension in degrees.
        - 'dec' : float
            Declination in degrees.
        - 'dRA' : float
            RA drift rate in arcseconds per minute.
        - 'dDec' : float
            Dec drift rate in arcseconds per minute.
    model_parameters : dict
        Dictionary containing pole coordinates:
        - 'alpha0' : float
            Reference right ascension (degrees).
        - 'delta0' : float
            Reference declination (degrees).
    synodic_period : float
        Synodic period in days.

    Returns
    -------
    tuple
        (sidereal_period, sidereal_period_alt, epoch1) where:
        - sidereal_period : float
            Sidereal period corresponding to the nominal pole.
        - sidereal_period_alt : float
            Sidereal period corresponding to the antipodal pole.
        - epoch1 : float
            Epoch of the first observation (CJD).
    """
    ra0 = model_parameters["alpha0"]
    dec0 = model_parameters["delta0"]

    epoch1 = data["cjd"].values[0]

    ra_t0 = data["ra"].values[0]
    dec_t0 = data["dec"].values[0]
    dRA = data["dRA"].values[0]
    dDec = data["dDec"].values[0]

    ra_t1, dec_t1 = (
        angle_after_one_synodic_period(ra_t0, synodic_period, dRA),
        angle_after_one_synodic_period(dec_t0, synodic_period, dDec),
    )

    atan_param_1 = calc_atan_parameter(ra_t0, dec_t0, ra0, dec0)
    atan_param_2 = calc_atan_parameter(ra_t1, dec_t1, ra0, dec0)

    ra_alt, dec_alt = flip_spin(ra0, dec0)

    atan_param_1alt = calc_atan_parameter(ra_t0, dec_t0, ra_alt, dec_alt)
    atan_param_2alt = calc_atan_parameter(ra_t1, dec_t1, ra_alt, dec_alt)

    sidereal_period = (
        2 * np.pi * synodic_period / (atan_param_2 - atan_param_1 + 2 * np.pi)
    )
    sidereal_period_alt = (
        2 * np.pi * synodic_period / (atan_param_2alt - atan_param_1alt + 2 * np.pi)
    )

    return sidereal_period, sidereal_period_alt, epoch1


def oblateness(a_b, a_c):
    """
    Compute a simple oblateness proxy from axis ratios.

    Parameters
    ----------
    a_b : float or array_like
        Intermediate-to-long axis ratio (a/b).
    a_c : float or array_like
        Short-to-long axis ratio (a/c).

    Returns
    -------
    float or array_like
        Oblateness
    """
    return 1 / 2 * a_b / a_c + 1 / 2 * 1 / a_b


def wrap_longitude(long):
    """
    Wrap a longitude angle into the range [0, 360).

    Parameters
    ----------
    long : float or array_like
        Input longitude in degrees.

    Returns
    -------
    float or array_like
        Longitude wrapped to the interval [0, 360).
    """
    return long % 360


def wrap_latitude(lat):
    """
    Wrap a latitude angle into the range [-90, 90] by folding across the poles.

    Parameters
    ----------
    lat : float or array_like
        Input latitude in degrees.

    Returns
    -------
    float or array_like
        Latitude wrapped to the interval [-90, 90].
    """
    m = (lat + 90) % 360  # shift so -90 maps to 0
    if m > 180:
        m = 360 - m
    return m - 90


def generate_initial_points(ra, dec, dec_shift=45):
    """
    Generate a set of 12 initial (RA, Dec) sampling points by combining
    spin-flipped poles, RA sweeps, and latitude shifts.

    Parameters
    ----------
    ra : float
        Base right ascension in degrees.
    dec : float
        Base declination in degrees.
    dec_shift : float, optional
        Latitude shift (degrees) applied when generating the secondary
        sets of points. Defaults to 45 degrees.

    Returns
    -------
    tuple of list
        (ra_list, dec_list), where each list contains 18 elements
        representing the generated right ascensions and declinations
        in degrees.
    """
    if np.abs(2 * dec - dec_shift) < 10:
        dec_shift += 20

    ra_list = []
    dec_list = []

    base_coords = [(ra, dec), flip_spin(ra, dec)]

    ra_sweep = [0, 180]

    for base_ra, base_dec in base_coords:
        for offset in ra_sweep:
            ra_list.append(wrap_longitude(base_ra + offset))
            dec_list.append(base_dec)

    dec_sweep = [-dec_shift, dec_shift]
    for shift in dec_sweep:
        for base_ra, base_dec in base_coords:
            if (base_dec + shift > 90) | (base_dec - shift < 90):
                flag = 1
            else:
                flag = 0
            shifted_dec = wrap_latitude(base_dec + shift)

            for offset in ra_sweep:
                temp_ra = base_ra + (180 if flag == 1 else 0)
                ra_list.append(wrap_longitude(temp_ra + offset))
                dec_list.append(shifted_dec)
    return ra_list, dec_list


def gaussian_interpolate(data, factor=4, sigma=1.0):
    """
    Reproduce matplotlib's `interpolation="gaussian"` effect.
    factor : upsampling factor
    sigma  : Gaussian smoothing strength
    """
    from scipy.ndimage import zoom, gaussian_filter

    # Step 1: upsample (mimics interpolation grid)
    up = zoom(data, factor, order=1)  # bilinear before smoothing

    # Step 2: apply gaussian smoothing
    smoothed = gaussian_filter(up, sigma=sigma)

    return smoothed


def detect_local_minima(arr):
    from scipy.ndimage import generate_binary_structure, minimum_filter, binary_erosion

    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    # https://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    neighborhood = generate_binary_structure(len(arr.shape), 2)

    footprint = np.ones((15, 15))
    local_min = minimum_filter(arr, footprint=footprint) == arr

    background = arr == 0
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )

    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)


def trumpet(peak_diff_1, f_feat, f_obs, f_2=None, kterm=1):
    """
    Scalar implementation of the alias/true flagging algorithm.

    Parameters
    ----------
    peak_diff_1 : float
        Difference between the 2nd and 3rd highest periodogram peaks.
    f_feat : float
        Feature frequency.
    f_obs : float
        Frequency of the highest peak in the periodogram.
    f_2 : float, optional
        Secondary frequency (harmonic / alias).
        Required if kterm == 2.
    kterm : int, optional
        Regime selector:
        - kterm == 1 : linear trumpet
        - kterm == 2 : curved trumpet

    Returns
    -------
    float
        Expected peak difference if the peak is true, otherwise 0.0.
    """

    # ------------------------------
    # Linear trumpet (original logic)
    # ------------------------------
    if kterm == 1:
        if peak_diff_1 > 0 and f_obs > f_feat:
            return 2.0 * f_feat

        elif peak_diff_1 < 0 and f_obs < f_feat:
            return -2.0 * f_obs

        elif peak_diff_1 > 0 and f_obs < f_feat:
            return 2.0 * f_obs

        elif peak_diff_1 < 0 and f_obs > f_feat:
            return -2.0 * f_feat

        return 0.0

    # ------------------------------
    # Curved trumpet (kterm == 2)
    # ------------------------------
    elif kterm == 2:
        if f_2 is None:
            raise ValueError("f_2 must be provided when kterm == 2")

        if f_2 > f_obs:
            return -0.5 * f_obs - f_feat

        elif f_2 < f_obs:
            return -0.5 * f_obs + f_feat

        return 0.0

    # ------------------------------
    # Fallback
    # ------------------------------
    return 0.0


def Nintervals(a):
    """
    Estimate the number of sampling intervals for period estimation
    as a function of semi major axis a.
    Parameters
    ----------
    a : float
        Semi major axis in AU.
    Returns
    -------
    int
        Estimated number of intervals.
    """
    Na = 71.073 * np.exp(-1.21 * a) + 2.528
    return int(Na)


def period_range(a, Psyn):
    """
    Compute the allowed period range around a reference synodic period.

    Parameters
    ----------
    a : float
        Semi major axis in AU.
    Psyn : float
        Reference synodic period in hours.
    Returns
    -------
    float
        Period range width in hours.
    """
    g = 1.619 * np.exp(-0.338 * a) - 5.069
    W = 10 ** (2 * np.log10(Psyn) + g)
    return W
