# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Part of this code is developped by Yongkai Zhu, modified by Chenxi Shan

__version__ = "1.2"
__date__    = "2022-08-28"

import os
import copy
import random
import pickle

import numpy as np
import pandas as pd

import sys
sys.path.append(r'../')

from utils.WilmanDB import *
from utils.spectrum import *
from utils.methods import *
from utils.functions import in_FoV
from utils.galaxy import RSgen

import matplotlib.pyplot as plt
from astropy import convolution
from astropy.io import fits as fits
import astropy.units as u
from radio_beam import Beam

from scipy.ndimage.interpolation import shift
from scipy.signal import fftconvolve

import timeit
from datetime import datetime

from skytools import GenSkyModel
from skytools import Source_Blending

class genBlendSky:
    """
    *** genBlendSky generate skymodel with appointed params ***
    It calls the GenSkyModel class & the Source_Blending class
    """
    
    def __init__(self, 
                 pix_arcsec=1, fov_deg=5,         # Sky frame base
                 flux_ratio=0.05, n_ratio=0.05,   # Blending flux & number contral
                 a_ratio=0.05,                    # Spectral error ratio
                 beam=4,                          # [arcsec] Blended beam size
                 sep=5,                           # [arcsec] Blended distance
                 auto=True,                       # Auto load DB & gen sky flag
                 host_db=None,                    # Host DB name
                 blend_bd=None,                   # Blend DB name
                 outdir = '../results/',          # Output dir
                 logging=False                    # Logger contral >> Add later
                ):
        
        # Sky frame
        self.pix_arcsec = pix_arcsec
        self.fov_deg = fov_deg
        self.PARAMS = self.gen_PARAMS( pix_arcsec=pix_arcsec, fov_deg=fov_deg )
        
        # Filename IO
        self.host_db_name = host_db
        self.blend_db_name = blend_db
        
        # Parsing params
        self.n_ratio = n_ratio
        self.flux_ratio = flux_ratio
        self.a_ratio = a_ratio
        self.beam = beam
        self.distance = sep
        self.outdir = outdir
        
        # Loading data
        if auto:
            self._preprocessing()
            self.simulate_blend_sky()
            print("After check the blended sky, you can finish the simulation by calling self.postprocessing")
            
        else:
            print("Please simulate the blend sky manually!")
    
    def _preprocessing( self ):
        self.load_DB()
        self.check_distance_beam_status()
        self.gen_defaut_freq_channels()
        self.gen_default_fnames()
        self.gen_host_blend_pair_indices()
        self.cal_host_blend_flux_ratio()
        
    def simulate_blend_sky( self ):
        self.gen_blend_pairs()
        self.filter_blend_pairs()
        self.convert_maps()
        self.visual_check_blend_pairs()
        self.add_ideal_sky()
        self.add_blend_sky()
    
    def postprocessing( self ):
        self.save_blend_pairs()
        self.sim_multifreq_ideal_sky( band='low' )
        self.sim_multifreq_blend_sky( band='low' )
        self.sim_multifreq_ideal_sky( band='mid' )
        self.sim_multifreq_blend_sky( band='mid' )
        self.sim_multifreq_ideal_sky( band='hig' )
        self.sim_multifreq_blend_sky( band='hig' )
    
    # ========================== Properties & Method ==========================
    """
    *** Properties & their methods of genBlendSky ***
    """ 
    
    @property
    def ideal_sky( self ):
        return self._ideal_sky
    
    def add_ideal_sky( self ):
        self._ideal_sky = self.skymodel_GMM_host_DB + self.ideal_blend
    
    @property
    def blend_sky( self ):
        return self._blend_sky
    
    def add_blend_sky( self ):
        self._blend_sky = self.ideal_unblended_host + self.blend_pairs
    
    # ========================== Blend Main Method ==========================
    """
    *** Main Methods for genBlendSky ***
    !!! Those methods are called by self.simulate_blend_sky() when auto is on !!!
    """ 
    
    def gen_blend_pairs( self ):
        """
        !!! calls gloabal variable skymodel_GMM_151, skymodel_GMM_151_2000 from load_DB() !!!
        *** calls cal_blend_position(), cal_blend_pair_alpha(), get_nonzero_values() ***
        :Output blended_skymodel:  Rescaled, convoluted blend;
        :Output blending_skymodel: Rescaled, convoluted blend + convoluted host;
        """
        # consider rename the names below?
        blending_skymodel = []   # Rescaled & convoluted blend + convoluted host;
        blended_skymodel = []    # Rescaled blend;
        
        for hi, si in zip(self.blend_host_indices, self.blend_sub_indices):
            blending_skymodel_i = {}
            skymodel1 = skymodel_GMM_151[hi]
            skymodel2 = skymodel_GMM_151_2000[si]
            
            SB = Source_Blending( beam2=self.beam )
            coord1, coord2 = self.cal_blend_position()
            T1 = SB.blend_models( skymodel1, skymodel2, coord1, coord2, self.r )
            blended_skymodel.append( SB.s2 )
            
            if SB.s1['type'] == 'FR1' or SB.s1['type'] == 'FR2':
                blending_skymodel_i['x'] = SB.s1['core_x']
                blending_skymodel_i['y'] = SB.s1['core_y']
                spectral_index = self.cal_blend_pair_alpha( SB.s1, SB.s2 )
            else:
                if SB.s2['type'] == 'FR1' or SB.s2['type'] == 'FR2':
                    spectral_index = self.cal_blend_pair_alpha( SB.s1, SB.s2 )
                else:
                    spectral_index = 0.70 + 0.70 * (2 * np.random.random() - 1) * self.a_ratio
                blending_skymodel_i['x'] = SB.s1['x']
                blending_skymodel_i['y'] = SB.s1['y']

            blending_skymodel_data = self.get_nonzero_values(T1)
            for hkey in skymodel1.keys():
                blending_skymodel_i['host_'+hkey] = skymodel1[hkey]
            for skey in skymodel2.keys():
                blending_skymodel_i['sub_'+skey] = skymodel2[skey]
            blending_skymodel_i['data'] = blending_skymodel_data
            blending_skymodel_i['spectral_index'] = spectral_index
            blending_skymodel_i['type'] = 'blended_source'
            blending_skymodel_i['relative_x'] = SB.s1['relative_x']
            blending_skymodel_i['relative_y'] = SB.s1['relative_y']
            blending_skymodel.append(blending_skymodel_i)
        
        self.blending_skymodel = blending_skymodel
        self.blend_pairs = blending_skymodel              # !!! Updated name
        self.blended_skymodel = blended_skymodel
        self.ideal_blend = blended_skymodel               # !!! Updated name
    
    def filter_blend_pairs( self ):
        """
        *** filter blend host from the 4000 source skymodel ***
        """
        unblended_sources = []
        for i in range(len(self.skymodel_GMM_host_DB)):
            if i not in self.blend_host_indices:
                unblended_sources.append(self.skymodel_GMM_host_DB[i])
        self.ideal_unblended_host = unblended_sources     # !!! Updated name
    
    def convert_maps( self ):
        """
        *** Convert sky model to images ***
        """
        # create basis
        img_size = int( self.fov_deg * 3600 / self.pix_arcsec )
        
        self.img_ideal_host_DB = self.skymodel_to_img_allsky( self.skymodel_GMM_host_DB, [img_size, img_size] )
        self.img_ideal_blend = self.skymodel_to_img_allsky( self.ideal_blend, [img_size, img_size] )
        self.img_ideal_unblend_host = self.skymodel_to_img_allsky( self.ideal_unblended_host, [img_size, img_size] )
        self.img_blend_pairs = self.skymodel_to_img_allsky( self.blend_pairs, [img_size, img_size] )
        self.img_ideal_sky = self.img_ideal_host_DB + self.img_ideal_blend
        self.img_blend_sky = self.img_ideal_unblend_host + self.img_blend_pairs
    
    def visual_check_blend_pairs( self ):
        """
        *** Visual inspect the blend pairs to view the difference. ***
        *** Calls plot_blend_pair() ***
        """
        for i in range(self.n_pair):
            idx = i + 1
            print("!!!=================== Inspecting pair # %s  ===================!!!" % idx)
            self.plot_blend_pair(i)
    
    def cal_freq_flux( self, skymodel, freq ):
        """
        *** cal_freq_flux() calculate flux of given freq from the 151 MHz skymodel ***

        :Params skymodel: 151 MHz skymodel from ();
        :Params freq: aimed freq;
        :Output total_flux: total_flux @ freq;
        """    
        source = copy.deepcopy(skymodel)
        if 'type' in source.keys():
            Type = source['type']
            if source['type'] == 'RQ':
                source['data'][:, 2] = rqq_spec(source['data'][:, 2], freq)
            if source['type'] == 'FR1':
                source['core_data'][:, 2] = fr1_core_spec(source['core_data'][:, 2], freq)
                source['lobe1_data'][:, 2] = fr1_lobe_spec(source['lobe1_data'][:, 2], freq)
                source['lobe2_data'][:, 2] = fr1_lobe_spec(source['lobe2_data'][:, 2], freq)
                source['data'] = np.vstack((source['core_data'], source['lobe1_data']))
                source['data'] = np.vstack((source['data'], source['lobe2_data']))
            if source['type'] == 'FR2':
                source['core_data'][:, 2] = fr2_core_spec(source['core_data'][:, 2], freq)
                source['lobe1_data'][:, 2] = fr2_lobe_spec(source['lobe1_data'][:, 2], freq)
                source['lobe2_data'][:, 2] = fr2_hotspot_spec(source['lobe2_data'][:, 2], freq)
                source['hotspot1_data'][:, 2] = fr2_hotspot_spec(source['hotspot1_data'][:, 2], freq)
                source['hotspot2_data'][:, 2] = fr2_hotspot_spec(source['hotspot2_data'][:, 2], freq)
                source['data'] = np.vstack((source['core_data'], source['lobe1_data']))
                source['data'] = np.vstack((source['data'], source['lobe2_data']))
                source['data'] = np.vstack((source['data'], source['hotspot1_data']))
                source['data'] = np.vstack((source['data'], source['hotspot2_data']))
            if source['type'] == 'SF':
                source['data'][:, 2] = sf_spec(source['data'][:, 2], freq)
            if source['type'] == 'SB':
                source['data'][:, 2] = sb_spec(source['data'][:, 2], freq)
        else:
            Type = source['sftype']
            if source['sftype'] == 4:
                source['data'][:, 2] = sf_spec(source['data'][:, 2], freq)
            if source['sftype'] == 5:
                source['data'][:, 2] = sb_spec(source['data'][:, 2], freq)
        total_flux = np.sum(source['data'][:, 2])
        print('Type %s | Total flux @ %s: %s' % (Type, freq, total_flux))
        return total_flux

    def cal_blend_pair_alpha( self, skymodel1, skymodel2 ):
        """
        *** cal_blended_alpha() calculate the blended spectral index ***
        !!! calls cal_freq_flux() !!!
        """
        I1_120 = self.cal_freq_flux(skymodel1, 120)
        I1_196 = self.cal_freq_flux(skymodel1, 196)
        I2_120 = self.cal_freq_flux(skymodel2, 120)
        I2_196 = self.cal_freq_flux(skymodel2, 196)
        I = {}
        I['120'] = I1_120 + I2_120
        I['196'] = I1_196 + I2_196
        ff = np.log10( 196 / 120 )
        II = np.log10( I['196'] / I['120'] )
        alpha = np.abs( II / ff )
        print('Calculated alpha: %s' % alpha)
        return alpha
    
    def check_distance_beam_status( self ):
        """
        *** Check if sep & beam size is compatible ***
        """
        npix_distance = int(self.distance/self.pix_arcsec)
        npix_beam = int(self.beam/self.pix_arcsec)
        if npix_distance > npix_beam:
            print("The separation is larger than the beam size, you might want to check if blend works.")
            
    def cal_blend_position( self ):
        """
        *** cal_blend_params() generate the distance & beamsize for the Source_Blending class ***
        """
        coord1 = [0, 0]
        
        npix_distance = int(self.distance/self.pix_arcsec)
        npix_beam = int(self.beam/self.pix_arcsec)
        sign = 1 if np.random.random() < 0.5 else -1
        srange_upper = npix_distance + 1 # Upper is excluded
        srange_lower = npix_beam / 2 + 1 # Lower is included
        s = sign * np.random.randint(srange_lower, srange_upper,1)
        if np.random.random() < 0.5:
            coord2=[0, s]
        else:
            coord2=[s, 0]
            
        return coord1, coord2
    
    # ========================== Multi-frequency Method ==========================
    """
    *** Multi-frequency Methods for genBlendSky ***
    !!! Those methods are called by self._postprocessing !!!
    """
    
    def sim_multifreq_ideal_sky( self, band='low' ):
        """
        *** Simulate & write multi_freqs ideal sky model in fits ***
        !!! Calls sim_multifreq() & writetofitsK() !!!
        """
        if band == 'low':
            print("Simulate low band of the idead sky.")
            low = self.sim_multifreq( self.ideal_sky, self.simulated_freqs_low )
            for f in low.keys():
                i = float(f)
                fn = self.ideal_fname.format(frequency=i)
                tic=timeit.default_timer()
                self.writetofitsK( fn, low[str(i)], i )
                toc=timeit.default_timer()
                print("Done writing freq %s MHz in time: %s" % (i, toc - tic) )
        elif band == 'mid':
            print("Simulate mid band of the idead sky.")
            mid = self.sim_multifreq( self.ideal_sky, self.simulated_freqs_mid )
            for f in mid.keys():
                i = float(f)
                fn = self.ideal_fname.format(frequency=i)
                tic=timeit.default_timer()
                self.writetofitsK( fn, mid[str(i)], i )
                toc=timeit.default_timer()
                print("Done writing freq %s MHz in time: %s" % (i, toc - tic) )
        elif band == 'hig':
            print("Simulate mid band of the idead sky.")
            hig = self.sim_multifreq( self.ideal_sky, self.simulated_freqs_hig )
            for f in hig.keys():
                i = float(f)
                fn = self.ideal_fname.format(frequency=i)
                tic=timeit.default_timer()
                self.writetofitsK( fn, hig[str(i)], i )
                toc=timeit.default_timer()
                print("Done writing freq %s MHz in time: %s" % (i, toc - tic) )
        
    def sim_multifreq_blend_sky( self, band='low' ):
        """
        *** Simulate & write multi_freqs blend sky model in fits ***
        !!! Calls sim_multifreq() & writetofitsK() !!!
        """
        if band == 'low':
            print("Simulate low band of the blend sky.")
            low = self.sim_multifreq( self.blend_sky, self.simulated_freqs_low )
            for f in low.keys():
                i = float(f)
                fn = self.blend_fname.format(frequency=i)
                tic=timeit.default_timer()
                self.writetofitsK( fn, low[str(i)], i )
                toc=timeit.default_timer()
                print("Done writing freq %s MHz in time: %s" % (i, toc - tic) )
        elif band == 'mid':
            print("Simulate mid band of the blend sky.")
            mid = self.sim_multifreq( self.blend_sky, self.simulated_freqs_mid )
            for f in mid.keys():
                i = float(f)
                fn = self.blend_fname.format(frequency=i)
                tic=timeit.default_timer()
                self.writetofitsK( fn, mid[str(i)], i )
                toc=timeit.default_timer()
                print("Done writing freq %s MHz in time: %s" % (i, toc - tic) )
        elif band == 'hig':
            print("Simulate mid band of the blend sky.")
            hig = self.sim_multifreq( self.blend_sky, self.simulated_freqs_hig )
            for f in hig.keys():
                i = float(f)
                fn = self.blend_fname.format(frequency=i)
                tic=timeit.default_timer()
                self.writetofitsK( fn, hig[str(i)], i )
                toc=timeit.default_timer()
                print("Done writing freq %s MHz in time: %s" % (i, toc - tic) )
    
    def sim_multifreq( self, inputmodel, simulated_freqs ):
        """
        *** sim_multifreq() generate multi-frequency skymodel from the 151 MHz skymodel ***
        !!! It calls methods from utils.spectrum & blend_pairs_spec() !!!
        
        :Params simulated_freqs: input ndarray;
        :Params skymodel: 151 MHz skymodel from Simulate();
        :Output multi_skymodel: multifrequency skymodel | dict;
        """    
        skymodel = copy.deepcopy(inputmodel)
        multi_skymodel = {}

        for freq_i in simulated_freqs:
            skymodel_i = []
            freq = freq_i #MHz
            for i in range(len(skymodel)):
                source = copy.deepcopy(skymodel[i])
                source['sim_freq'] = freq
                if 'type' in source.keys():
                    if source['type'] == 'RQ':
                        source['data'][:, 2] = rqq_spec(source['data'][:, 2], freq)
                    if source['type'] == 'FR1':
                        source['core_data'][:, 2] = fr1_core_spec(source['core_data'][:, 2], freq)
                        source['lobe1_data'][:, 2] = fr1_lobe_spec(source['lobe1_data'][:, 2], freq)
                        source['lobe2_data'][:, 2] = fr1_lobe_spec(source['lobe2_data'][:, 2], freq)
                        source['data'] = np.vstack((source['core_data'], source['lobe1_data']))
                        source['data'] = np.vstack((source['data'], source['lobe2_data']))
                    if source['type'] == 'FR2':
                        source['core_data'][:, 2] = fr2_core_spec(source['core_data'][:, 2], freq)
                        source['lobe1_data'][:, 2] = fr2_lobe_spec(source['lobe1_data'][:, 2], freq)
                        source['lobe2_data'][:, 2] = fr2_hotspot_spec(source['lobe2_data'][:, 2], freq)
                        source['hotspot1_data'][:, 2] = fr2_hotspot_spec(source['hotspot1_data'][:, 2], freq)
                        source['hotspot2_data'][:, 2] = fr2_hotspot_spec(source['hotspot2_data'][:, 2], freq)
                        source['data'] = np.vstack((source['core_data'], source['lobe1_data']))
                        source['data'] = np.vstack((source['data'], source['lobe2_data']))
                        source['data'] = np.vstack((source['data'], source['hotspot1_data']))
                        source['data'] = np.vstack((source['data'], source['hotspot2_data']))
                    if source['type'] == 'SF':
                        source['data'][:, 2] = sf_spec(source['data'][:, 2], freq)
                    if source['type'] == 'SB':
                        source['data'][:, 2] = sb_spec(source['data'][:, 2], freq)
                    if source['type'] == 'blended_source':
                        source['data'][:, 2] = self.blend_pairs_spec(source['data'][:, 2], freq, source['spectral_index'])
                else:
                    if source['sftype'] == 4:
                        source['data'][:, 2] = sf_spec(source['data'][:, 2], freq)
                    if source['sftype'] == 5:
                        source['data'][:, 2] = sb_spec(source['data'][:, 2], freq)
                skymodel_i.append(source)
            multi_skymodel[str(freq_i)] = skymodel_i

        return multi_skymodel

    def blend_pairs_spec( self, I_151, freqMHz, index):
        return (freqMHz / 151.0) ** (-1 * np.abs(index)) * I_151
    
    # ========================== Preprocessing Method ==========================
    """
    *** Preprocessing Methods for genBlendSky ***
    !!! Those methods are called by self._preprocessing() when auto is on !!!
    """
    
    def load_DB( self ):
        print("+++++++++++ LOADING WILMAN DB +++++++++++")
        self.WDB = self.load_WilmanDB('../data')
        
        print("+++++++++++ LOADING HOSTS DB +++++++++++")
        if self.host_db_name is None:
            print("No host db input, loading default host DB")
            self.skymodel_GMM_host_DB = self.read_skymodel('skymodel_GMM_151_s4000_2022-08-18_fixfluxerror.pkl')
        else:
            self.skymodel_GMM_host_DB = self.read_skymodel(self.host_db_name)
        self.n_host_db = len(self.skymodel_GMM_host_DB)
        print("Host database have %s sources." % self.n_host_db)
        
        print("+++++++++++ LOADING BLENDS DB +++++++++++")
        if self.blend_db_name is None:
            print("No host db input, loading default host DB")
            self.skymodel_GMM_blend_DB = self.read_skymodel('skymodel_GMM_151_s2000_2022-08-18_fixfluxerror.pkl')
        else:
            self.skymodel_GMM_blend_DB = self.read_skymodel(self.blend_db_name)
        self.n_blend_db = len(self.skymodel_GMM_blend_DB)
        print("Blend database have %s sources." % self.n_blend_db)
        
        print("++++++++++ Calculate # of BLEND Pairs +++++++++++")
        self.n_pair =  int( self.n_host_db * self.n_ratio )
        print("Number of blend pairs is set to %s" % self._npair)
        # Set global variables
        global skymodel_GMM_151, skymodel_GMM_151_2000
        skymodel_GMM_151 = self.skymodel_GMM_host_DB
        skymodel_GMM_151_2000 = self.skymodel_GMM_blend_DB
        
    def gen_defaut_freq_channels( self ):
        """
        *** Gen the default frequency channels ***
        """
        self.simulated_freqs_low = np.linspace(120, 128, 51)
        self.simulated_freqs_mid = np.linspace(154, 162, 51)
        self.simulated_freqs_hig = np.linspace(188, 196, 51)
        
    def gen_default_fnames( self ):
        """
        *** Gen defualt file name for multi-feqs ideal sky & blend sky ***
        """
        self.namehandle = '_n_pairs_' + str(self.n_pair) + '_f_ratio_' + str(self.flux_ratio).replace(".", "") + '_a_ratio_' + str(self.a_ratio).replace(".", "")
        self.ideal_fname = self.outdir + 'ideal/' + "Idealsky_{frequency:06.2f}" + self.namehandle + ".fits"
        self.blend_fname = self.outdir + 'blend/' + "Blendsky_{frequency:06.2f}" + self.namehandle + ".fits"
        
    def gen_host_blend_pair_indices( self ):
        """
        *** Gen the host & blend pair ***
        """
        size = self.n_pair
        
        blend_host_indices = np.random.randint(0, high=len(self.skymodel_GMM_host_DB), size=size, dtype=int)
        n_blend_host_indices = len(set(blend_host_indices))
        blend_host_indices = list(set(blend_host_indices))
        
        while n_blend_host_indices < size:
            ind = np.random.randint(0, high=len(self.skymodel_GMM_host_DB), size=1, dtype=int)
            if ind not in blend_host_indices:
                blend_host_indices.append(ind[0])
                n_blend_host_indices += 1
        
        blend_sub_indices = np.random.randint(0, high=len(self.skymodel_GMM_blend_DB), size=size, dtype=int)
        if size >= 10:
            blend_sub_indices[0:4] = [2, 4, 20, 99] # Include extended sources?
        n_blend_sub_indices = len(set(blend_sub_indices))
        blend_sub_indices = list(set(blend_sub_indices))
        
        while n_blend_sub_indices < size:
            ind = np.random.randint(0, high=len(self.skymodel_GMM_blend_DB), size=1, dtype=int)
            if ind not in blend_sub_indices:
                blend_sub_indices.append(ind[0])
                n_blend_sub_indices += 1
                
        print(blend_host_indices, blend_sub_indices)
        self.blend_host_indices = blend_host_indices
        self.blend_sub_indices = blend_sub_indices
    
    def cal_host_blend_flux_ratio( self ):
        """
        !!! calls gloabal variable skymodel_GMM_151, skymodel_GMM_151_2000 from load_DB() !!!
        """
        tot_flux_s4000_151 = 0
        for i in skymodel_GMM_151:
            tot_flux_s4000_151 += i['i_151_tot']
        self.tot_flux_s4000_151 = tot_flux_s4000_151

        tot_flux_sub_151 = 0
        for i in self.blend_sub_indices:
            tot_flux_sub_151 += skymodel_GMM_151_2000[i]['i_151_tot']
        self.tot_flux_sub_151 = tot_flux_sub_151

        r = tot_flux_s4000_151 * self.flux_ratio / tot_flux_sub_151
        # This is the adjustment to the blend flux ratio
        self.blend_flux_rescale_ratio = r
        self.r = r

        print('tot_flux_s4000_151:', tot_flux_s4000_151)
        print('tot_flux_sub200:', tot_flux_sub_151)
        print('ratio:', tot_flux_sub_151 / tot_flux_s4000_151, r)
    
    # ========================== I/O Method ==========================
    """
    *** Data or file I/O Methods ***
    """
    
    def load_WilmanDB( self, dbpath ):
        tic=timeit.default_timer()
        WDB = WilmanDB(dbpath=dbpath)
        toc=timeit.default_timer()
        print("Loading time:", toc - tic)
        return WDB
    
    def read_skymodel( self, fn ):
        tic=timeit.default_timer()
        pickle_open = open(fn, 'rb')
        skymodel_151 = pickle.load(pickle_open)
        pickle_open.close()
        toc=timeit.default_timer()
        print("Loading time:", toc - tic)
        return skymodel_151

    def save_skymodel( self, fn, skymodel ):
        tic=timeit.default_timer()
        pickle_skymodel = open( fn, 'wb' )
        pickle.dump( skymodel, pickle_skymodel )
        pickle_skymodel.close()
        toc=timeit.default_timer()
        print("Dumping time:", toc - tic)
    
    def save_blend_pairs( self ):
        """
        *** save the blend paris using save_skymodel() ***
        """
        d = datetime.now()
        dd = d.strftime('%m_%d_%Y_%H%M%S')
        blend_pairs_name = 'blended_data_151' + self.namehandle + '_' + dd + '.pkl'
        blend_pairs_data = [self.blend_host_indices, self.blend_sub_indices, self.blended_skymodel, self.blending_skymodel]
        self.save_skymodel(blend_pairs_name, blend_pairs_data)
    
    def writetofitsK( self, fn, all_sky_model, freq ):
        """
        *** writetofitsK() output fits image file ***
        !!! Calls calc_Tb() from utils.methods !!!
        """
        img_size = int( self.fov_deg * 3600 / self.pix_arcsec )
        img_all_sky = self.skymodel_to_img_allsky(all_sky_model, [img_size, img_size])
        JytoTb = calc_Tb(1, self.PARAMS['pix_area'], freq)
        img_all_sky_K = img_all_sky * JytoTb
        fitsname = fn
        header = fits.Header()
        header['IMGSIZE'] = img_all_sky.shape[0]
        print(fitsname)
        fits.writeto(fitsname, data=img_all_sky_K, header=header, overwrite=True)
    
    # ========================== Supplement Method ==========================
    """
    *** Supplement Method surves Supplement function ***
    """
    
    def get_nonzero_values(self, arr, lim=0):
        indices = np.nonzero(arr > lim)
        return np.transpose((indices[0], indices[1], arr[indices]))

    def normalize(self, arr, flux):
        return arr * flux / np.sum(arr)
    
    def gen_PARAMS( self, pix_arcsec=1, fov_deg=5 ):
        """
        *** gen_PARAMS() output PARAMS for GenSkyModel class  ***
        """
        pix_arcsec = pix_arcsec                                    # [arcsec] skymap pixel size
        fov_deg = fov_deg                                          # [deg] skymap FoV coverage
        #simulated_freqs = np.linspace(f_start, f_end, channel)    # [MHz] simulated frequency channels
        # This is not used by any method in GenSkyModel

        fov_arcsec = fov_deg * 3600.0         
        img_size = int(fov_arcsec / pix_arcsec)
        pix_size_deg = fov_deg / img_size
        pix_size_arcsec = fov_arcsec / img_size
        pix_area_arcsec2 = pix_size_arcsec ** 2
        ra_min = -1 * fov_deg / 2.0
        dec_min = -1 * fov_deg / 2.0

        PARAMS = {
            "img_size": img_size,
            "fov_deg": fov_deg,
            "fov_arcsec": fov_arcsec,
            "pix_deg": pix_size_deg,
            "pix_size": pix_size_arcsec,
            "pix_area": pix_area_arcsec2,
            "ra_min": ra_min,
            "dec_min": dec_min,
            "dmin": 100, # The minimum distance between sources. Its unit is pixel.
            "number_of_rq": 200*2,
            "number_of_fr1": 95*2,
            "number_of_fr2": 5*2,
            "number_of_sf": 1200*2,
            "number_of_sb": 500*2,
            "frequency": 158,
            "minimum_flux": 1.7536435940177778e-07, # Jy
            "maximum_flux": 17000,
        }
        ## If you don't want to simulate some kind of galaxy, just set the number of it to 0.
        print("Field of View: %.2f deg\nPixel Size: %.2f arcsec\nImage Size: %d" %(fov_deg, pix_size_arcsec, img_size))
        #print("Simulated Frequencies:\n", simulated_freqs)
        #check_mem(img_size)

        return PARAMS
    
    # ========================== Visualization Method ==========================
    """
    *** Visualization Method generate maps ***
    """
    
    def skymodel_to_img_allsky( self, skymodel, img_size ):
        """
        *** Convert skymodel to image for a large sky ***
        """
        w, h = img_size
        img = np.zeros([w, h])
        for source in skymodel:
            if 'type' in source.keys():
                if source['type'] == 'FR1' or source['type'] == 'FR2':
                    x = source['data'][:, 0] + source['core_x'] - source['relative_x']
                    y = source['data'][:, 1] + source['core_y'] - source['relative_y']
                else:
                    x = source['data'][:, 0] + source['x'] - source['relative_x']
                    y = source['data'][:, 1] + source['y'] - source['relative_y']
            else:
                x = source['data'][:, 0] + source['x'] - source['relative_x']
                y = source['data'][:, 1] + source['y'] - source['relative_y']
            for i in range(len(x)):
                img[int(x[i]), int(y[i])] += source['data'][i, 2]

        return img

    def skymodel_to_img_single_source( self, skymodel, img_size ):
        """
        *** Convert one skymodel to image ***
        """
        w, h = img_size
        img = np.zeros([w, h])
        x = skymodel[:, 0]
        y = skymodel[:, 1]
        for i in range(len(x)):
            img[int(x[i]), int(y[i])] += skymodel[i, 2]
        
        return img
    
    def plot_blend_pair( self, idx, canvas=30 ):
        """
        *** Plot blend pair based on idx of the pair ***
        """
        img1 = self.img_ideal_host_DB
        img2 = self.img_ideal_blend
        img_blend_pairs = self.img_blend_pairs
        img_ideal_sky = self.img_ideal_sky

        x = self.blend_pairs[idx]['x']
        y = self.blend_pairs[idx]['y']
        xleft = int( x - canvas )
        xright = int( x + canvas )
        yleft = int( y - canvas )
        yright = int( y + canvas )
        print(xleft,xright, yleft,yright)
        
        plt.figure(figsize=[18, 18])
        plt.subplot(331)
        plt.gca().set_title('Ideal host | footprint')
        plt.imshow(img1[xleft: xright, yleft: yright]>0)
        plt.subplot(332)
        plt.gca().set_title('Ideal blend | footprint')
        plt.imshow(img2[xleft: xright, yleft: yright]>0)
        plt.subplot(333)
        plt.gca().set_title('Ideal pair | footprint')
        plt.imshow(img_ideal_sky[xleft: xright, yleft: yright]>0)
        plt.subplot(334)
        plt.gca().set_title('Ideal host')
        plt.imshow(img1[xleft: xright, yleft: yright])
        plt.subplot(335)
        plt.gca().set_title('Ideal blend')
        plt.imshow(img2[xleft: xright, yleft: yright])
        plt.subplot(336)
        plt.gca().set_title('Ideal pair')
        plt.imshow(img_ideal_sky[xleft: xright, yleft: yright])
        plt.subplot(338)
        plt.gca().set_title('Blended pair | footprint')
        plt.imshow(img_blend_pairs[xleft: xright, yleft: yright]>0)
        plt.subplot(339)
        plt.gca().set_title('Blended pair')
        plt.imshow(img_blend_pairs[xleft: xright, yleft: yright])