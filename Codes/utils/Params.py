# Copyright (c) 2021-2022 Chenxi Shan <cxshan@hey.com>

from methods input check_mem

"""
The Default Params:

{'img_size': 14400,
 'fov_deg': 2,
 'fov_arcsec': 7200.0,
 'pix_deg': 0.0001388888888888889,
 'pix_size': 0.5,
 'pix_area': 0.25,
 'ra_min': -1.0,
 'dec_min': -1.0,
 'dmin': 12.0,
 'number_of_fr2': 5,
 'number_of_fr1': 95,
 'number_of_rq': 200,
 'number_of_sf': 1200,
 'number_of_sb': 500,
 'flux_frequency': 158,
 'minimum_flux': 1e-07,
 'maximum_flux': 17,
 'simulated_freqs': array([162.  , 161.84, 161.68, 161.52, 161.36, 161.2 , 161.04, 160.88,
        160.72, 160.56, 160.4 , 160.24, 160.08, 159.92, 159.76, 159.6 ,
        159.44, 159.28, 159.12, 158.96, 158.8 , 158.64, 158.48, 158.32,
        158.16, 158.  , 157.84, 157.68, 157.52, 157.36, 157.2 , 157.04,
        156.88, 156.72, 156.56, 156.4 , 156.24, 156.08, 155.92, 155.76,
        155.6 , 155.44, 155.28, 155.12, 154.96, 154.8 , 154.64, 154.48,
        154.32, 154.16, 154.  ])}
        
"""

def print_Params( Params ):
    """
    *** print_Params() prints each keys & results ***
    """
    for key in Params:
        print(key, ' : ', Params[key])
        
def gen_Params( pix_size=0.5, 
                fov_deg=2, 
                freqlist=[154, 162, 51],
                nlist=[5, 95, 200, 1200, 500],
                fluxlist=[1e-7,17,158],
                dmin = 6 # arcsec
              ):
    """
    *** gen_Params() for the Simulate Class ***
    
    Params freqlist: # of simulated frequencies in [min, max, channels] order;
    Params nlist: # of source in types = ['fr2', 'fr1', 'rq', 'sf', 'sb'] order;
    Params fluxlist: # of simulated flux bin in given flux_freq in [min, max, flux_freq] order;
    """
    # Image & pixel sizes
    pix_size_arcsec = pix_size
    fov_deg = fov_deg
    
    fov_arcsec = fov_deg * 3600.0
    img_size = int(fov_arcsec / pix_size_arcsec)
    pix_size_deg = fov_deg / img_size
    pix_area_arcsec2 = pix_size_arcsec ** 2
    
    ra_min = -1 * fov_deg / 2.0
    dec_min = -1 * fov_deg / 2.0
    
    # Freqs
    if len(freqlist) == 3:
        freq_min = freqlist[0]
        freq_max = freqlist[1]
        freq_channels = freqlist[2]
        simulated_freqs = np.linspace(freq_max, freq_min, freq_channels)
    else:
        print('Expect len(freqlist) == 3, getting len(freqlist) =', len(freqlist))
        raise ValueError('Please check the freqlist.')
    
    # Number of sources
    if len(nlist) == 5:
        n_fr2 = nlist[0]
        n_fr1 = nlist[1]
        n_rq = nlist[2]
        n_sf = nlist[3]
        n_sb = nlist[4]
    else:
        print('Expect len(nlist) == 5, getting len(nlist) =', len(nlist))
        raise ValueError('Please assign the # of all the types.')
    
    # Flux limits
    if len(fluxlist) == 3:
        flux_min = fluxlist[0]
        flux_max = fluxlist[1]
        flux_freq = fluxlist[2]
    else:
        print('Expect len(fluxlist) == 3, getting len(fluxlist) =', len(fluxlist))
        raise ValueError('Please check the fluxlist.')
    
    # Distance limit
    dmin_arcsec = dmin
    dmin_pixel = dmin_arcsec / pix_size_arcsec
    
    Params = {
    "img_size": img_size,
    "fov_deg": fov_deg,
    "fov_arcsec": fov_arcsec,
    "pix_deg": pix_size_deg,
    "pix_size": pix_size_arcsec,
    "pix_area": pix_area_arcsec2,
    "ra_min": ra_min,
    "dec_min": dec_min,
    "dmin": dmin_pixel, # The minimum distance between sources. Its unit is pixel.
    "number_of_fr2": n_fr2,
    "number_of_fr1": n_fr1,
    "number_of_rq": n_rq,
    "number_of_sf": n_sf,
    "number_of_sb": n_sb,
    "flux_frequency": flux_freq, # The frequency to calculate flux limit
    "minimum_flux": flux_min, # Jy
    "maximum_flux": flux_max, # Jy
    "simulated_freqs": simulated_freqs
    }
    
    
    print_Params( Params )
    print("============== Checking Memory ==============")
    check_mem(img_size)
    print("============== Checking Memory ==============")
    return Params