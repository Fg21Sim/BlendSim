# Copyright (c) 2021,2024 Yongkai Zhu <yongkai_zhu@hotmail.com>
# MIT License

"""
Functions to calculate the flux density at the given frequency.
Parameters:
    I_151: Flux density at 151 MHz
    freqMHz: Frequency of interest
"""

import numpy as np

def rqq_spec(I_151, freqMHz):
    """
    Radio spectrum of radio quiet galaxy.
    """
    return ((freqMHz / 151.0) ** (-0.7)) * I_151

def fr1_lobe_spec(I_151, freqMHz):
    """
    Radio spectrum of FR I lobe.
    """
    return ((freqMHz / 151.0) ** (-0.75)) * I_151

def fr1_core_spec(I_151, freqMHz):
    """
    Radio spectrum of the core of FR I.
    """
    freq = freqMHz * 1e6     # [Hz]
    a0 = (np.log10(I_151) - 0.07 * np.log10(151E6) +
          0.29 * np.log10(151E6) * np.log10(151E6))
    lgS = (a0 + 0.07 * np.log10(freq) -
           0.29 * np.log10(freq) * np.log10(freq))
    return 10.0 ** lgS

def fr2_core_spec(I_151, freqMHz):
    """
    Radio spectrum of the core of FR II.
    """
    freq = freqMHz * 1e6    # [Hz]
    a0 = (np.log10(I_151) - 0.07 * np.log10(151E6) +
          0.29 * np.log10(151E6) * np.log10(151E6))
    lgS = (a0 + 0.07 * np.log10(freq) -
           0.29 * np.log10(freq) * np.log10(freq))
    return 10.0 ** lgS

def fr2_lobe_spec(I_151, freqMHz):
    """
    Radio spectrum of the lobe of FR II.
    """
    return (freqMHz / 151.0) ** (-0.75) * I_151

def fr2_hotspot_spec(I_151, freqMHz):
    """
    Radio spectrum of the hotspot of FR II.
    """
    return (freqMHz / 151.0) ** (-0.75) * I_151

def sf_spec(I_151, freqMHz):
    """
    Radio spectrum of star forming galaxy.
    """
    return (freqMHz / 151.0) ** (-0.7) * I_151

def sb_spec(I_151,freqMHz):
    """
    Radio spectrum of star burst galaxy.
    """
    return (freqMHz / 151.0) ** (-0.7) * I_151

