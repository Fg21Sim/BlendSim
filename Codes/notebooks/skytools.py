# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Part of this code is mainly developped by Yongkai Zhu, modified by Chenxi Shan

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
from utils.methods import skymodel_to_img
from utils.methods import check_mem
import utils.methods as mthd
from utils.functions import in_FoV


from utils.galaxy import RSgen

import matplotlib.pyplot as plt


from astropy import convolution
from astropy.io import fits as fits
import astropy.units as u
from radio_beam import Beam

from scipy.ndimage.interpolation import shift
from scipy.signal import fftconvolve

def get_nonzero_values(arr, lim=0):
    indices = np.nonzero(arr > lim)
    return np.transpose((indices[0], indices[1], arr[indices]))

def normalize(arr, flux):
    return arr * flux / np.sum(arr)

class GenSkyModel:
    def __init__(self, PARAMS):
        self.PARAMS = PARAMS
    
    def fr1(self, source, model='GMM', Bmajor=1):
        if model == "GMM":
            skymodel = self.fr1_GMM(source, Bmajor=Bmajor)
        if model == "EllipseModel":
            skymodel = self.fr1_ell_model(source)
        return skymodel
    
    def fr1_ell_model(self, source):
        pass

    def fr1_GMM(self, source, Bmajor=1):
        source_info = self.get_fr1_info(source)
        lobe_minor_beam = source_info['lobe1_minor']
        lobe1_xy = [source_info['lobe1_x'], source_info['lobe1_y']]
        lobe2_xy = [source_info['lobe2_x'], source_info['lobe2_y']]
        core_xy = [source_info['core_x'], source_info['core_y']]
        if not in_FoV(0, self.PARAMS['img_size'], np.array(lobe1_xy), np.array(lobe2_xy), np.array(core_xy)):
            return None
        angle = source_info['lobe1_pa']
        core_major_beam = Bmajor
        core_beam = Beam(core_major_beam*u.arcsec, core_major_beam*u.arcsec, 0*u.deg)
        lobel_beam = Beam(lobe_minor_beam*u.arcsec, lobe_minor_beam*u.arcsec, 0*u.deg)
        a = source_info['lobe1_a']
        b = source_info['lobe1_b']
        pix_scale = self.PARAMS['pix_size']
        core_beam_kern_x_size = int(Bmajor / self.PARAMS['pix_size']) * 2 + 1
        core_beam_kern = core_beam.as_kernel(pix_scale*u.arcsec, x_size=core_beam_kern_x_size, y_size=core_beam_kern_x_size)
        lobe_beam_kern = lobel_beam.as_kernel(pix_scale*u.arcsec, x_size=int(b)*4+1, y_size=int(b)*4+1)
        padding_x = int(np.max([lobe_beam_kern.array.shape[0] / 2, core_beam_kern.array.shape[0]]))
        padding_y = int(np.max([lobe_beam_kern.array.shape[1] / 2, core_beam_kern.array.shape[1]]))
        int_lobe1_x, int_lobe1_y, t_lobe1, v_lobe1 = self.jet_radial_profile(lobe1_xy, a, angle)
        int_lobe2_x, int_lobe2_y, t_lobe2, v_lobe2 = self.jet_radial_profile(lobe2_xy, a, angle)
        xmin = np.min([int_lobe1_x.min(), int_lobe2_x.min()])
        ymin = np.min([int_lobe1_y.min(), int_lobe2_y.min()])
        int_lobe1_x_shifted = np.rint(int_lobe1_x -  xmin)
        int_lobe1_y_shifted = np.rint(int_lobe1_y - ymin)
        int_lobe2_x_shifted = np.rint(int_lobe2_x - xmin)
        int_lobe2_y_shifted = np.rint(int_lobe2_y - ymin)
        core_x_shifted = core_xy[0] - xmin
        core_y_shifted = core_xy[1] - ymin
        img_h = int(np.max([int_lobe1_x_shifted.max(), int_lobe2_x_shifted.max()]) + 1)
        img_w = int(np.max([int_lobe1_y_shifted.max(), int_lobe2_y_shifted.max()]) + 1)
        img = np.zeros([img_h, img_w])
        img = np.pad(img, [(padding_x, padding_x), (padding_y, padding_y)], mode='constant', constant_values=0)
        temp_core = img.copy()
        core_x_shifted = int(np.around(core_x_shifted + padding_x))
        core_y_shifted = int(np.around(core_y_shifted + padding_y))
        int_lobe1_x_shifted += padding_x
        int_lobe1_y_shifted += padding_y
        int_lobe2_x_shifted += padding_x
        int_lobe2_y_shifted += padding_y
        temp_core[core_x_shifted, core_y_shifted] = source_info['core_i_151']
        temp_lobe1 = img.copy()
        temp_lobe1[int_lobe1_x_shifted.astype(int), int_lobe1_y_shifted.astype(int)] = v_lobe1 * source_info['lobe1_i_151']
        temp_lobe2 = img.copy()
        temp_lobe2[int_lobe2_x_shifted.astype(int), int_lobe2_y_shifted.astype(int)] = v_lobe2 * source_info['lobe2_i_151']
        temp_core = convolution.convolve(temp_core, core_beam_kern.array, normalize_kernel=True)
        #temp_core = fftconvolve(temp_core, core_beam_kern.array, mode='same')
        temp_core[temp_core<1e-15] = 0
        temp_core = normalize(temp_core, source_info['core_i_151'])
        temp_core_nonzeros = get_nonzero_values(temp_core)
        temp_core_nonzeros[:, 2] = temp_core_nonzeros[:, 2] * source_info['core_i_151'] / np.sum(temp_core_nonzeros[:, 2])

        source_info['core_data'] = temp_core_nonzeros
        temp_lobe1 = convolution.convolve(temp_lobe1, lobe_beam_kern.array, normalize_kernel=True)
        temp_lobe1[temp_lobe1<1e-15] = 0
        temp_lobe1 = normalize(temp_lobe1, source_info['lobe1_i_151'])
        #temp_lobe1 = fftconvolve(temp_lobe1, lobe_beam_kern.array, mode='same')
        
        temp_lobe1_nonzeros = get_nonzero_values(temp_lobe1)
        temp_lobe1_nonzeros[:, 2] = temp_lobe1_nonzeros[:, 2] * source_info['lobe1_i_151'] / np.sum(temp_lobe1_nonzeros[:, 2])

        source_info['lobe1_data'] = temp_lobe1_nonzeros
        temp_lobe2 = convolution.convolve(temp_lobe2, lobe_beam_kern.array, normalize_kernel=True)
        temp_lobe2[temp_lobe2<1e-15] = 0
        temp_lobe2 = normalize(temp_lobe2, source_info['lobe2_i_151'])
        #temp_lobe2 = fftconvolve(temp_lobe2, lobe_beam_kern.array, mode='same')
        temp_lobe2_nonzeros = get_nonzero_values(temp_lobe2)
        temp_lobe2_nonzeros[:, 2] = temp_lobe2_nonzeros[:, 2] * source_info['lobe2_i_151'] / np.sum(temp_lobe2_nonzeros[:, 2])

        source_info['lobe2_data'] = temp_lobe2_nonzeros
        model = temp_core + temp_lobe1 + temp_lobe2
        source_info['data'] = get_nonzero_values(model)
        source_info['type'] = 'FR1' 
        source_info['relative_x'] = core_x_shifted
        source_info['relative_y'] = core_y_shifted
        x_shifted = core_xy[0] - core_x_shifted
        y_shifted = core_xy[1] - core_y_shifted
        if not in_FoV(0, self.PARAMS['img_size'], temp_core_nonzeros[:, 0]+x_shifted, temp_lobe1_nonzeros[:, 0]+x_shifted, temp_lobe2_nonzeros[:, 0]+x_shifted, temp_core_nonzeros[:, 1]+y_shifted, temp_lobe1_nonzeros[:, 1]+y_shifted, temp_lobe2_nonzeros[:, 1]+y_shifted):
            return None
        return source_info
    
    def fr2(self, source, model='GMM', Bmajor=1):
        if model == "GMM":
            skymodel = self.fr2_GMM(source, Bmajor=Bmajor)
        if model == "EllipseModel":
            skymodel = self.fr2_ell_model(source)
        return skymodel
    
    def fr2_ell_model(self, source):
        pass
    
    def fr2_GMM(self, source, Bmajor=1):
        source_info = self.get_fr2_info(source)
        lobe_minor_beam = source_info['lobe1_minor']
        lobe1_xy = [source_info['lobe1_x'], source_info['lobe1_y']]
        lobe2_xy = [source_info['lobe2_x'], source_info['lobe2_y']]
        core_xy = [source_info['core_x'], source_info['core_y']]
        hp1_xy = [source_info['hp1_x'], source_info['hp1_y']]
        hp2_xy = [source_info['hp2_x'], source_info['hp2_y']]
        if not in_FoV(0, self.PARAMS['img_size'], np.array(lobe1_xy), np.array(lobe2_xy), np.array(core_xy), np.array(hp1_xy), np.array(hp2_xy)):
            return None
        angle = source_info['lobe1_pa']
        core_major_beam = Bmajor
        core_beam = Beam(core_major_beam*u.arcsec, core_major_beam*u.arcsec, 0*u.deg)
        lobel_beam = Beam(lobe_minor_beam*u.arcsec, lobe_minor_beam*u.arcsec, 0*u.deg)
        a = source_info['lobe1_a']
        b = source_info['lobe1_b']
        pix_scale = self.PARAMS['pix_size']
        core_beam_kern_x_size = int(Bmajor / self.PARAMS['pix_size']) * 2 + 1
        core_beam_kern = core_beam.as_kernel(pix_scale*u.arcsec, x_size=core_beam_kern_x_size, y_size=core_beam_kern_x_size)
        lobe_beam_kern = lobel_beam.as_kernel(pix_scale*u.arcsec, x_size=int(b)*4+1, y_size=int(b)*4+1)
        padding_x = int(lobe_beam_kern.array.shape[0] / 2)
        padding_y = int(lobe_beam_kern.array.shape[1] / 2)
        int_lobe1_x, int_lobe1_y, t_lobe1, v_lobe1 = self.jet_radial_profile(lobe1_xy, a, angle)
        int_lobe2_x, int_lobe2_y, t_lobe2, v_lobe2 = self.jet_radial_profile(lobe2_xy, a, angle)
        xmin = np.min([int_lobe1_x.min(), int_lobe2_x.min(), hp1_xy[0], hp2_xy[0]])
        ymin = np.min([int_lobe1_y.min(), int_lobe2_y.min(), hp1_xy[1], hp2_xy[1]])
        int_lobe1_x_shifted = np.rint(int_lobe1_x -  xmin)
        int_lobe1_y_shifted = np.rint(int_lobe1_y - ymin)
        int_lobe2_x_shifted = np.rint(int_lobe2_x - xmin)
        int_lobe2_y_shifted = np.rint(int_lobe2_y - ymin)
        core_x_shifted = np.rint(core_xy[0] - xmin)
        core_y_shifted = np.rint(core_xy[1] - ymin)
        hp1_x_shifted = np.rint(hp1_xy[0] - xmin)
        hp1_y_shifted = np.rint(hp1_xy[1] - ymin)
        hp2_x_shifted = np.rint(hp2_xy[0] - xmin)
        hp2_y_shifted = np.rint(hp2_xy[1] - ymin)
        img_h = int(np.max([int_lobe1_x_shifted.max(), int_lobe2_x_shifted.max(), hp1_x_shifted, hp2_x_shifted]) + 1)
        img_w = int(np.max([int_lobe1_y_shifted.max(), int_lobe2_y_shifted.max(), hp1_y_shifted, hp2_y_shifted]) + 1)
        img = np.zeros([img_h, img_w])
        img = np.pad(img, [(padding_x, padding_x), (padding_y, padding_y)], mode='constant', constant_values=0)
        temp_core = img.copy()
        temp_hp1 = img.copy()
        temp_hp2 = img.copy()
        core_x_shifted = int(np.around(core_x_shifted + padding_x))
        core_y_shifted = int(np.around(core_y_shifted + padding_y))
        hp1_x_shifted = int(np.around(hp1_x_shifted + padding_x))
        hp1_y_shifted = int(np.around(hp1_y_shifted + padding_y))
        hp2_x_shifted = int(np.around(hp2_x_shifted + padding_x))
        hp2_y_shifted = int(np.around(hp2_y_shifted + padding_y))
        int_lobe1_x_shifted += padding_x
        int_lobe1_y_shifted += padding_y
        int_lobe2_x_shifted += padding_x
        int_lobe2_y_shifted += padding_y
        temp_core[core_x_shifted, core_y_shifted] = source_info['core_i_151']
        temp_hp1[hp1_x_shifted, hp1_y_shifted] = source_info['hotspot1_i_151']
        temp_hp2[hp2_x_shifted, hp2_y_shifted] = source_info['hotspot2_i_151']
        temp_lobe1 = img.copy()
        temp_lobe1[int_lobe1_x_shifted.astype(int), int_lobe1_y_shifted.astype(int)] = v_lobe1 * source_info['lobe1_i_151']
        temp_lobe2 = img.copy()
        temp_lobe2[int_lobe2_x_shifted.astype(int), int_lobe2_y_shifted.astype(int)] = v_lobe2 * source_info['lobe2_i_151']
        temp_core = convolution.convolve(temp_core, core_beam_kern.array, normalize_kernel=True)
        temp_core[temp_core<1e-15] = 0
        temp_core = normalize(temp_core, source_info['core_i_151'])
        temp_core_nonzeros = get_nonzero_values(temp_core)
        #temp_core_nonzeros[:, 2] = temp_core_nonzeros[:, 2] * source_info['core_i_151'] / np.sum(temp_core_nonzeros[:, 2])
        source_info['core_data'] = temp_core_nonzeros
        
        temp_hp1 = convolution.convolve(temp_hp1, core_beam_kern.array, normalize_kernel=True)
        temp_hp1[temp_hp1<1e-15] = 0
        temp_hp1 = normalize(temp_hp1, source_info['hotspot1_i_151'])
        temp_hp1_nonzeros = get_nonzero_values(temp_hp1)
        #temp_hp1_nonzeros[:, 2] = temp_hp1_nonzeros[:, 2] * source_info['hotspot1_i_151'] / np.sum(temp_hp1_nonzeros[:, 2])
        source_info['hotspot1_data'] = temp_hp1_nonzeros
        
        temp_hp2 = convolution.convolve(temp_hp2, core_beam_kern.array, normalize_kernel=True)
        temp_hp2[temp_hp2<1e-15] = 0
        temp_hp2 = normalize(temp_hp2, source_info['hotspot2_i_151'])
        temp_hp2_nonzeros = get_nonzero_values(temp_hp2)
        #temp_hp2_nonzeros[:, 2] = temp_hp2_nonzeros[:, 2] * source_info['hotspot2_i_151'] / np.sum(temp_hp2_nonzeros[:, 2])
        source_info['hotspot2_data'] = temp_hp2_nonzeros
        
        temp_lobe1 = convolution.convolve(temp_lobe1, lobe_beam_kern.array, normalize_kernel=True)
        temp_lobe1[temp_lobe1<1e-15] = 0
        temp_lobe1 = normalize(temp_lobe1, source_info['lobe1_i_151'])
        temp_lobe1_nonzeros = get_nonzero_values(temp_lobe1)
        #temp_lobe1_nonzeros[:, 2] = temp_lobe1_nonzeros[:, 2] * source_info['lobe1_i_151'] / np.sum(temp_lobe1_nonzeros[:, 2])
        source_info['lobe1_data'] = temp_lobe1_nonzeros
        
        temp_lobe2 = convolution.convolve(temp_lobe2, lobe_beam_kern.array, normalize_kernel=True)
        temp_lobe2[temp_lobe2<1e-15] = 0
        temp_lobe2 = normalize(temp_lobe2, source_info['lobe2_i_151'])
        temp_lobe2_nonzeros = get_nonzero_values(temp_lobe2)
        #temp_lobe2_nonzeros[:, 2] = temp_lobe2_nonzeros[:, 2] * source_info['lobe2_i_151'] / np.sum(temp_lobe2_nonzeros[:, 2])
        source_info['lobe2_data'] = temp_lobe2_nonzeros
        
        source_info['type'] = 'FR2'
        source_info['relative_x'] = core_x_shifted
        source_info['relative_y'] = core_y_shifted
        x_shifted = core_xy[0] - core_x_shifted
        y_shifted = core_xy[1] - core_y_shifted
        if not in_FoV(0, self.PARAMS['img_size'], temp_core_nonzeros[:, 0]+x_shifted, source_info['hotspot1_data'][:, 0]+x_shifted, source_info['hotspot2_data'][:, 0]+x_shifted, source_info['lobe1_data'][:, 0]+x_shifted, source_info['lobe2_data'][:, 0]+x_shifted, temp_core_nonzeros[:, 1]+y_shifted, source_info['hotspot1_data'][:, 1]+y_shifted, source_info['hotspot2_data'][:, 1]+y_shifted, source_info['lobe1_data'][:, 1]+y_shifted, source_info['lobe2_data'][:, 1]+y_shifted):
            return None
        model = temp_core + temp_hp1 + temp_hp2 + temp_lobe1 + temp_lobe2
        source_info['data'] = get_nonzero_values(model)
        return source_info
                
    def get_fr1_info(self, source):
        source_info = {}
        index = source.index
        source_info['index'] = source.index
        source_info['galaxy'] = source.loc[index[0]].galaxy
        source_info['agntype'] = source.loc[index[0]].agntype
        source_info['i_151_tot'] = source.loc[index[0]].i_151_tot
        source_info['redshift'] = source.loc[index[0]].redshift
        #core
        source_info['core_structure'] = source.loc[index[0]].structure
        source_info['core_ra'] = source.loc[index[0]].ra - self.PARAMS['ra_min']
        source_info['core_dec'] = source.loc[index[0]].dec - self.PARAMS['dec_min']
        source_info['core_i_151'] = source.loc[index[0]].i_151_flux
        #lobel
        source_info['lobe1_structure'] = source.loc[index[1]].structure
        source_info['lobe1_ra'] = source.loc[index[1]].ra - self.PARAMS['ra_min']
        source_info['lobe1_dec'] = source.loc[index[1]].dec - self.PARAMS['dec_min']
        source_info['lobe1_i_151'] = source.loc[index[1]].i_151_flux
        source_info['lobe1_pa'] = source.loc[index[1]].pa
        source_info['lobe1_major'] = source.loc[index[1], 'major_axis']
        source_info['lobe1_minor'] = source.loc[index[1], 'minor_axis']
        #lobe2
        source_info['lobe2_structure'] = source.loc[index[2]].structure
        source_info['lobe2_ra'] = source.loc[index[2]].ra - self.PARAMS['ra_min']
        source_info['lobe2_dec'] = source.loc[index[2]].dec - self.PARAMS['dec_min']
        source_info['lobe2_i_151'] = source.loc[index[2]].i_151_flux
        source_info['lobe2_pa'] = source.loc[index[2]].pa
        source_info['lobe2_major'] = source.loc[index[2], 'major_axis']
        source_info['lobe2_minor'] = source.loc[index[2], 'minor_axis']
        core_x, lobe1_x, lobe2_x = (source.ra - self.PARAMS['ra_min']) / self.PARAMS['pix_deg']
        core_y, lobe1_y, lobe2_y = (source.dec - self.PARAMS['dec_min']) / self.PARAMS['pix_deg']
        source_info['core_x'] = core_x
        source_info['core_y'] = core_y
        source_info['lobe1_x'] = lobe1_x
        source_info['lobe1_y'] = lobe1_y
        source_info['lobe2_x'] = lobe2_x
        source_info['lobe2_y'] = lobe2_y   
        a1 = 0.5 * source_info['lobe1_major'] / self.PARAMS['pix_size']
        b1 = 0.5 * source_info['lobe1_minor'] / self.PARAMS['pix_size']
        source_info['lobe1_a'] = a1
        source_info['lobe1_b'] = b1
        a2 = 0.5 * source_info['lobe2_major'] / self.PARAMS['pix_size']
        b2 = 0.5 * source_info['lobe2_minor'] / self.PARAMS['pix_size']
        source_info['lobe2_a'] = a2
        source_info['lobe2_b'] = b2
        return source_info
        
    def get_fr2_info(self, source):
        source_info = {}
        index = source.index
        source_info['index'] = source.index
        source_info['galaxy'] = source.loc[index[0]].galaxy
        source_info['agntype'] = source.loc[index[0]].agntype
        source_info['i_151_tot'] = source.loc[index[0]].i_151_tot
        source_info['redshift'] = source.loc[index[0]].redshift
        #core
        source_info['core_structure'] = source.loc[index[0]].structure
        source_info['core_ra'] = source.loc[index[0]].ra - self.PARAMS['ra_min']
        source_info['core_dec'] = source.loc[index[0]].dec - self.PARAMS['dec_min']
        source_info['core_i_151'] = source.loc[index[0]].i_151_flux
        #lobel
        source_info['lobe1_structure'] = source.loc[index[1]].structure
        source_info['lobe1_ra'] = source.loc[index[1]].ra - self.PARAMS['ra_min']
        source_info['lobe1_dec'] = source.loc[index[1]].dec - self.PARAMS['dec_min']
        source_info['lobe1_i_151'] = source.loc[index[1]].i_151_flux
        source_info['lobe1_pa'] = source.loc[index[1]].pa
        source_info['lobe1_major'] = source.loc[index[1], 'major_axis']
        source_info['lobe1_minor'] = source.loc[index[1], 'minor_axis']
        #lobe2
        source_info['lobe2_structure'] = source.loc[index[2]].structure
        source_info['lobe2_ra'] = source.loc[index[2]].ra - self.PARAMS['ra_min']
        source_info['lobe2_dec'] = source.loc[index[2]].dec - self.PARAMS['dec_min']
        source_info['lobe2_i_151'] = source.loc[index[2]].i_151_flux
        source_info['lobe2_pa'] = source.loc[index[2]].pa
        source_info['lobe2_major'] = source.loc[index[2], 'major_axis']
        source_info['lobe2_minor'] = source.loc[index[2], 'minor_axis']
        #hotspot1
        source_info['hotspot1_ra'] = source.loc[index[3]].ra - self.PARAMS['ra_min']
        source_info['hotspot1_dec'] = source.loc[index[3]].dec - self.PARAMS['dec_min']
        source_info['hotspot1_i_151'] = source.loc[index[3]].i_151_flux
        source_info['hotspot1_structure'] = source.loc[index[3]].structure
        #hotspot2
        source_info['hotspot2_ra'] = source.loc[index[4]].ra - self.PARAMS['ra_min']
        source_info['hotspot2_dec'] = source.loc[index[4]].dec - self.PARAMS['dec_min']
        source_info['hotspot2_i_151'] = source.loc[index[4]].i_151_flux
        source_info['hotspot2_structure'] = source.loc[index[4]].structure    

        core_x, lobe1_x, lobe2_x, hp1_x, hp2_x = (source.ra - self.PARAMS['ra_min']) / self.PARAMS['pix_deg']
        core_y, lobe1_y, lobe2_y, hp1_y, hp2_y = (source.dec - self.PARAMS['dec_min']) / self.PARAMS['pix_deg']
        
        source_info['core_x'] = core_x
        source_info['core_y'] = core_y
        source_info['lobe1_x'] = lobe1_x
        source_info['lobe1_y'] = lobe1_y
        source_info['lobe2_x'] = lobe2_x
        source_info['lobe2_y'] = lobe2_y   
        source_info['hp1_x'] = hp1_x
        source_info['hp1_y'] = hp1_y
        source_info['hp2_x'] = hp2_x
        source_info['hp2_y'] = hp2_y
        
        a1 = 0.5 * source_info['lobe1_major'] / self.PARAMS['pix_size']
        b1 = 0.5 * source_info['lobe1_minor'] / self.PARAMS['pix_size']
        source_info['lobe1_a'] = a1
        source_info['lobe1_b'] = b1
        a2 = 0.5 * source_info['lobe2_major'] / self.PARAMS['pix_size']
        b2 = 0.5 * source_info['lobe2_minor'] / self.PARAMS['pix_size']
        source_info['lobe2_a'] = a2
        source_info['lobe2_b'] = b2
        return source_info
            
    def sfsb(self, source, model='GMM', Bmajor=1):
        if model == "GMM":
            skymodel = self.sfsb_GMM(source, Bmajor=Bmajor)
        if model == "EllipseModel":
            skymodel = self.sfsb_ell_model(source)
        return skymodel
        
    
    def sfsb_ell_model(self, source):
        """
        Star-formation / starburst galaxies

        """  
        source_info = self.get_sfsb_info(source)
        x = source_info['x']
        y = source_info['y']
        if in_FoV(0, self.PARAMS['img_size'], x, y):
            a = 0.5 * source_info['major'] / self.PARAMS['pix_size']
            b = 0.5 * source_info['minor'] / self.PARAMS['pix_size']
            xmin = int(np.round(x - a))
            xmax = int(np.round(x + a))
            ymin = int(np.round(y - b))
            ymax = int(np.round(y + b))
            xc = x - xmin
            yc = y - ymin
            pix_info = []
            if xc == 0:
                xc = 1
            if yc == 0:
                yc = 1
            if a == 0:
                a = 1
            if b == 0:
                b = 1
            ellipse = mthd.draw_ellipse([2*xc, 2*yc], [xc,yc], a, b, source_info['pa'])
            area = np.sum(ellipse)
            if area == 0:
                area = 1
            if area == 1:
                pix = (xc, yc, source_info['i_151'])
                pix_info.append(pix)
                source_info['sim_freq'] = 151
                source_info['relative_x'] = x
                source_info['relative_y'] = x
                source_info['data'] = np.array(pix_info)
                if source_info['sftype'] == 4:
                    source_info['type'] = 'SF'
                if source_info['sftype'] == 5:
                    source_info['type'] = 'SB'
            else:
                flux_pix = source_info['i_151'] / area
                ellipse = ellipse * flux_pix
                ellipse_nonzero_values = get_nonzero_values(ellipse)
                x_min = np.min(ellipse_nonzero_values[:, 0])
                y_min = np.min(ellipse_nonzero_values[:, 1])
                ellipse_nonzero_values[:, 0] -= x_min
                ellipse_nonzero_values[:, 1] -= y_min
                source_info['data'] = ellipse_nonzero_values
                source_info['relative_x'] = xc - x_min
                source_info['relative_y'] = yc - y_min
                x_shifted = x - source_info['relative_x']
                y_shifted = y - source_info['relative_x']
                if in_FoV(0, self.PARAMS['img_size'], ellipse_nonzero_values[:, 0]+x_shifted, ellipse_nonzero_values[:, 1]+y_shifted):
                    if source_info['sftype'] == 4:
                        source_info['type'] = 'SF'
                    if source_info['sftype'] == 5:
                        source_info['type'] = 'SB'
                else:
                    return None
        else:
            return None
        return source_info
        
    def sfsb_GMM(self, source, Bmajor=1):
        """
        Star-formation / starburst galaxies

        """  
        source_info = self.get_sfsb_info(source)
        x = source_info['x']
        y = source_info['y']
        if in_FoV(0, self.PARAMS['img_size'], x, y):
            a = 0.5 * source_info['major'] / self.PARAMS['pix_size']
            b = 0.5 * source_info['minor'] / self.PARAMS['pix_size']
            xmin = int(np.round(x - a))
            xmax = int(np.round(x + a))
            ymin = int(np.round(y - b))
            ymax = int(np.round(y + b))
            xc = x - xmin
            yc = y - ymin
            pix_info = []
            if xc == 0:
                xc = 1
            if yc == 0:
                yc = 1
            if a == 0:
                a = 1
            if b == 0:
                b = 1
            ellipse = mthd.draw_ellipse([2*xc, 2*yc], [xc,yc], a, b, source_info['pa'])
            core_beam = Beam(Bmajor*u.arcsec, Bmajor*u.arcsec, 0*u.deg)
            core_beam_kern_x_size = int(Bmajor / self.PARAMS['pix_size']) * 2 + 1
            core_beam_kern = core_beam.as_kernel(self.PARAMS['pix_size']*u.arcsec, x_size=core_beam_kern_x_size,y_size=core_beam_kern_x_size)
            padding_x = int(core_beam_kern.shape[0] / 2)
            padding_y = int(core_beam_kern.shape[1] / 2)
            area = np.sum(ellipse)
            if area == 0:
                area = 1
            if area == 1:
                core_x_shifted = 0
                core_y_shifted = 0
                temp_core = np.zeros([1, 1])
                temp_core = np.pad(temp_core, [(padding_x, padding_x), (padding_y, padding_y)], mode='constant', constant_values=0)
                core_x_shifted = int(np.around(core_x_shifted + padding_x))
                core_y_shifted = int(np.around(core_y_shifted + padding_y))
                temp_core[core_x_shifted, core_y_shifted] = source_info['i_151']
                temp_core = convolution.convolve(temp_core, core_beam_kern.array, normalize_kernel=True)
                temp_core[temp_core<1e-15] = 0
                temp_core = normalize(temp_core, source_info['i_151'])
                temp_core_nonzeros = get_nonzero_values(temp_core)
                source_info['relative_x'] = core_x_shifted
                source_info['relative_y'] = core_y_shifted
                source_info['data'] = temp_core_nonzeros
                source_info['sim_freq'] = 151
                x_shifted = x - source_info['relative_x']
                y_shifted = y - source_info['relative_x']
                if source_info['sftype'] == 4:
                    source_info['type'] = 'SF'
                if source_info['sftype'] == 5:
                    source_info['type'] = 'SB'
                if not in_FoV(0, self.PARAMS['img_size'], temp_core_nonzeros[:, 0]+x_shifted, temp_core_nonzeros[:, 1]+y_shifted):
                    return None
            else:
                flux_pix = source_info['i_151'] / area
                ellipse = ellipse * flux_pix
                ellipse_padded = np.pad(ellipse, [(padding_x, padding_x), (padding_y, padding_y)], mode='constant', constant_values=0)
                ellipse_conv = convolution.convolve(ellipse_padded, core_beam_kern.array, normalize_kernel=True)
                ellipse_conv[ellipse_conv<1e-15] = 0
                ellipse_conv = normalize(ellipse_conv, source_info['i_151'])
                ellipse_conv_nonzeros = get_nonzero_values(ellipse_conv)
                x_min = np.min(ellipse_conv_nonzeros[:, 0])
                y_min = np.min(ellipse_conv_nonzeros[:, 1])
                ellipse_conv_nonzeros[:, 0] -= x_min
                ellipse_conv_nonzeros[:, 1] -= y_min
                source_info['relative_x'] = xc + padding_x - x_min
                source_info['relative_y'] = yc + padding_y - y_min
                source_info['data'] = ellipse_conv_nonzeros
                source_info['sim_freq'] = 151
                x_shifted = x - source_info['relative_x']
                y_shifted = y - source_info['relative_x']
                if in_FoV(0, self.PARAMS['img_size'], ellipse_conv_nonzeros[:, 0]+x_shifted, ellipse_conv_nonzeros[:, 1]+y_shifted):
                    if source_info['sftype'] == 4:
                        source_info['type'] = 'SF'
                    if source_info['sftype'] == 5:
                        source_info['type'] = 'SB'
                else:
                    return None
        else:
            return None
        return source_info
        
    def get_sfsb_info(self, source):
        source_info = {}
        source_info['index'] = source.index[0]
        gindex = source.index[0]
        source_info['galaxy'] = source.loc[gindex, 'galaxy'] # galaxy number
        source_info['sftype'] = source.loc[gindex, 'sftype'] + 3 # galaxy type;
        source_info['i_151_tot'] = source.loc[gindex, 'i_151_tot']
        source_info['redshift'] = source.loc[gindex, 'redshift']
        source_info['structure'] = source.loc[gindex, 'structure']
        source_info['ra'] = source.loc[gindex, 'ra'] - self.PARAMS['ra_min']
        source_info['dec'] = source.loc[gindex, 'dec'] - self.PARAMS['dec_min']
        source_info['i_151'] = source.loc[gindex, 'i_151_flux']
        source_info['pa'] = source.loc[gindex, 'pa']
        source_info['major'] = source.loc[gindex, 'major_axis']
        source_info['minor'] = source.loc[gindex, 'minor_axis']
        source_info['x'] = int(source_info['ra'] / self.PARAMS['pix_deg'])
        source_info['y'] = int(source_info['dec'] / self.PARAMS['pix_deg'])
        return source_info
        
    
    def rq(self, source, model='GMM', Bmajor=1):
        if model == "GMM":
            skymodel = self.rq_GMM(source, Bmajor=Bmajor)
        if model == "PointModel":
            skymodel = self.rq_point_model(source)
        return skymodel
    
    def rq_point_model(self, source):
        """
        Radio Quiet galaxies.

        """
        source_info = self.get_rq_info(source)
        source_info['type'] = 'RQ'
        x = source_info['x']
        y = source_info['y']
        if not in_FoV(0, self.PARAMS['img_size'], x, y):
            return None
        pix_info = [(0, 0, source_info['i_151'])]
        source_info['data'] = np.array(pix_info)
        source_info['sim_freq'] = 151
        source_info['relative_x'] = 0
        source_info['relative_y'] = 0
        return source_info
    
    def rq_GMM(self, source, Bmajor=1):
        """
        Radio Quiet galaxies.

        """  
        source_info = self.get_rq_info(source)
        x = source_info['x']
        y = source_info['y']
        core_x_shifted = 0
        core_y_shifted = 0
        core_beam = Beam(Bmajor*u.arcsec, Bmajor*u.arcsec, 0*u.deg)
        core_beam_kern_x_size = int(Bmajor / self.PARAMS['pix_size']) * 2 + 1
        core_beam_kern = core_beam.as_kernel(self.PARAMS['pix_size']*u.arcsec, x_size=core_beam_kern_x_size,y_size=core_beam_kern_x_size)
        padding_x = int(core_beam_kern.shape[0] / 2)
        padding_y = int(core_beam_kern.shape[1] / 2)
        temp_core = np.zeros([1, 1])
        temp_core = np.pad(temp_core, [(padding_x, padding_x), (padding_y, padding_y)], mode='constant', constant_values=0)
        core_x_shifted = int(np.around(core_x_shifted + padding_x))
        core_y_shifted = int(np.around(core_y_shifted + padding_y))
        temp_core[core_x_shifted, core_y_shifted] = source_info['i_151']
        temp_core = convolution.convolve(temp_core, core_beam_kern.array, normalize_kernel=True)
        temp_core[temp_core<1e-15] = 0
        temp_core = normalize(temp_core, source_info['i_151'])
        temp_core_nonzeros = get_nonzero_values(temp_core)
        source_info['relative_x'] = core_x_shifted
        source_info['relative_y'] = core_y_shifted
        source_info['data'] = temp_core_nonzeros
        source_info['sim_freq'] = 151
        x_shifted = x - source_info['relative_x']
        y_shifted = y - source_info['relative_x']
        if not in_FoV(0, self.PARAMS['img_size'], temp_core_nonzeros[:, 0]+x_shifted, temp_core_nonzeros[:, 1]+y_shifted):
            return None
        return source_info
    
    def get_rq_info(self, source):
        source_info = {}
        source_info['type'] = 'RQ'
        source_info['index'] = source.index[0]
        gindex = source.index[0]
        source_info['i_151'] = source.loc[gindex, 'i_151_flux']
        source_info['ra'] = source.loc[gindex, 'ra'] - self.PARAMS['ra_min']
        source_info['dec'] = source.loc[gindex, 'dec'] - self.PARAMS['dec_min']
        source_info['redshift'] = source.loc[gindex, 'redshift']
        source_info['galaxy'] = source.loc[gindex, 'galaxy']
        source_info['agntype'] = source.loc[gindex, 'agntype']
        source_info['structure'] = source.loc[gindex, 'structure']
        source_info['i_151_tot'] = source.loc[gindex, 'i_151_tot']
        source_info['x'] = int(source_info['ra'] / self.PARAMS['pix_deg'])
        source_info['y'] = int(source_info['dec'] / self.PARAMS['pix_deg'])
        return source_info
    
    def jet_radial_profile(self, mu, sig, angle):
        a = sig * 2
        t = np.arange(-a, a+1, 1)
        x, y = self.linear_function(mu, t, angle)
        int_x, int_y = np.rint(x), np.rint(y)
        unique_index = abs(int_x[1:] - int_x[:-1]) + abs(int_y[1:] - int_y[:-1])
        unique_index = unique_index.astype(bool)
        unique_index = np.insert(unique_index, 0, values=1, axis=0)
        int_x = int_x[unique_index]
        int_y = int_y[unique_index]
        #new_t = (int_x - mu[0]) ** 2 + (int_y - mu[0]) ** 2
        new_t = t[unique_index]
        flux = 1 / sig /np.sqrt(2 * np.pi) * np.exp(-np.power(new_t - 0, 2.) / (2 * np.power(sig, 2.)))
        flux = flux / np.sum(flux)
        return [int_x, int_y, new_t, flux]
    
    def linear_function(self, center, t, theta):
        x = center[0] + t * np.sin(theta)
        y = center[1] + t * np.cos(theta)
        return [x, y]
    
class Source_Blending:
    """
    *** Source Blending of EDRS ***
    """
    def __init__(self, image_size=400, pix_scale=1, beam1=2, beam2=4):
        self.Beam1 = Beam(beam1*u.arcsec, beam1*u.arcsec, 0*u.deg)
        self.Beam2 = Beam(beam2*u.arcsec, beam2*u.arcsec, 0*u.deg)
        self.Beam_Beam12Beam2 = self.Beam2.deconvolve(self.Beam1)
        self.image_size = image_size
        self.pix_scale = pix_scale * u.arcsec
        self.kernel_beam1 = self.Beam1.as_kernel(self.pix_scale)
        self.kernel_beam2 = self.Beam2.as_kernel(self.pix_scale)
        kern_size = beam2 / pix_scale * 4 + 1
        self.kernel_beam12beam2 = self.Beam_Beam12Beam2.as_kernel(self.pix_scale, x_size=kern_size, y_size=kern_size)
    
    def skymodel_to_img_single_source(self, skymodel, img_size):
        w, h = img_size
        img = np.zeros([w, h])
        x = skymodel[:, 0]
        y = skymodel[:, 1]
        for i in range(len(x)):
            img[int(x[i]), int(y[i])] += skymodel[i, 2]

        return img

    def blend_models(self, skymodel1, skymodel2, d1, d2, r):
        d1x, d1y = d1
        d2x, d2y = d2
        self.s1 = copy.deepcopy(skymodel1)
        self.s2 = copy.deepcopy(skymodel2)
        s1_img_w = int(np.max(self.s1['data'][:, 0]))
        s1_img_h = int(np.max(self.s1['data'][:, 1]))
        s2_img_w = int(np.max(self.s2['data'][:, 0]))
        s2_img_h = int(np.max(self.s2['data'][:, 1]))
        s1_cent_x = self.s1['relative_x']
        s1_cent_y = self.s1['relative_y']
        s2_cent_x = self.s2['relative_x']
        s2_cent_y = self.s2['relative_y']
        #kern_half_size = int(self.kernel_beam12beam2.array.shape[0] / 2)
        kern_size = int(self.kernel_beam12beam2.array.shape[0])
        dest_img_w = int((s1_img_w + s2_img_w + np.sqrt(d1x**2+d1y**2)*2 + kern_size)/2) * 2 + 1
        dest_img_h = int((s1_img_h + s2_img_h + np.sqrt(d2x**2+d2y**2)*2 + kern_size)/2) * 2 + 1
        img = np.zeros([dest_img_w, dest_img_h])
        x1 = int(dest_img_w / 2 - s1_cent_x)
        y1 = int(dest_img_h / 2 - s1_cent_y)
        x2 = int(dest_img_w / 2 - s2_cent_x)
        y2 = int(dest_img_h / 2 - s2_cent_y)
        self.s1['data'][:, 0] += x1
        self.s1['data'][:, 1] += y1
        self.s2['data'][:, 0] += x2
        self.s2['data'][:, 1] += y2 
        self.s2['data'][:, 2] = self.s2['data'][:, 2] * r
        self.s2['i_151_tot'] = self.s2['i_151_tot'] * r
        if self.s2['type'] == 'FR1':
            self.s2['core_i_151'] = self.s2['core_i_151'] * r
            self.s2['lobe1_i_151'] = self.s2['lobe1_i_151'] * r
            self.s2['lobe2_i_151'] = self.s2['lobe2_i_151'] * r
            self.s2['core_data'][:, 0] += x2 + d2x
            self.s2['lobe1_data'][:, 0] += x2 + d2x
            self.s2['lobe2_data'][:, 0] += x2 + d2x
            self.s2['core_data'][:, 1] += y2 + d2y
            self.s2['lobe1_data'][:, 1] += y2 + d2y
            self.s2['lobe2_data'][:, 1] += y2 + d2y
        elif self.s2['type'] == 'FR2':
            self.s2['core_i_151'] = self.s2['core_i_151'] * r
            self.s2['lobe1_i_151'] = self.s2['lobe1_i_151'] * r
            self.s2['lobe2_i_151'] = self.s2['lobe2_i_151'] * r
            self.s2['hotspot1_i_151'] = self.s2['hotspot1_i_151'] * r
            self.s2['hotspot2_i_151'] = self.s2['hotspot2_i_151'] * r
            self.s2['core_data'][:, 0] += x2 + d2x
            self.s2['lobe1_data'][:, 0] += x2 + d2x
            self.s2['lobe2_data'][:, 0] += x2 + d2x
            self.s2['hotspot1_data'][:, 0] += x2 + d2x
            self.s2['hotspot2_data'][:, 0] += x2 + d2x
            self.s2['core_data'][:, 1] += y2 + d2y
            self.s2['lobe1_data'][:, 1] += y2 + d2y
            self.s2['lobe2_data'][:, 1] += y2 + d2y
            self.s2['hotspot1_data'][:, 1] += y2 + d2y
            self.s2['hotspot2_data'][:, 1] += y2 + d2y
        else:
            self.s2['i_151'] = self.s2['i_151'] * r
            
        if self.s1['type'] == 'FR1':
            self.s1['core_i_151'] = self.s1['core_i_151'] * r
            self.s1['lobe1_i_151'] = self.s1['lobe1_i_151'] * r
            self.s1['lobe2_i_151'] = self.s1['lobe2_i_151'] * r
            self.s1['core_data'][:, 0] += x1 + d1x
            self.s1['lobe1_data'][:, 0] += x1 + d1x
            self.s1['lobe2_data'][:, 0] += x1 + d1x
            self.s1['core_data'][:, 1] += y1 + d1y
            self.s1['lobe1_data'][:, 1] += y1 + d1y
            self.s1['lobe2_data'][:, 1] += y1 + d1y
        if self.s1['type'] == 'FR2':
            self.s1['core_i_151'] = self.s1['core_i_151'] * r
            self.s1['lobe1_i_151'] = self.s1['lobe1_i_151'] * r
            self.s1['lobe2_i_151'] = self.s1['lobe2_i_151'] * r
            self.s1['hotspot1_i_151'] = self.s1['hotspot1_i_151'] * r
            self.s1['hotspot2_i_151'] = self.s1['hotspot2_i_151'] * r
            self.s1['core_data'][:, 0] += x1 + d1x
            self.s1['lobe1_data'][:, 0] += x1 + d1x
            self.s1['lobe2_data'][:, 0] += x1 + d1x
            self.s1['hotspot1_data'][:, 0] += x1 + d1x
            self.s1['hotspot2_data'][:, 0] += x1 + d1x
            self.s1['core_data'][:, 1] += y1 + d1y
            self.s1['lobe1_data'][:, 1] += y1 + d1y
            self.s1['lobe2_data'][:, 1] += y1 + d1y
            self.s1['hotspot1_data'][:, 1] += y1 + d1y
            self.s1['hotspot2_data'][:, 1] += y1 + d1y
            
        self.template1 = self.skymodel_to_img_single_source(self.s1['data'], [dest_img_w, dest_img_h])
        self.template2 = self.skymodel_to_img_single_source(self.s2['data'], [dest_img_w, dest_img_h])
        self.template1_conv_beam2 = convolution.convolve(self.template1, self.kernel_beam12beam2, normalize_kernel=True)
        self.template1_conv_beam2 = shift(self.template1_conv_beam2, [d1x, d1y], cval=0)
        self.template1_conv_beam2[self.template1_conv_beam2<1e-15] = 0
        self.template1_conv_beam2 = normalize(self.template1_conv_beam2, self.s1['i_151_tot'])
        self.template2_conv_beam2 = convolution.convolve(self.template2, self.kernel_beam12beam2, normalize_kernel=True)
        self.template2_conv_beam2 = shift(self.template2_conv_beam2, [d2x, d2y], cval=0)
        self.template2_conv_beam2[self.template2_conv_beam2<1e-15] = 0
        self.template2_conv_beam2 = normalize(self.template2_conv_beam2, self.s2['i_151_tot'])
        self.s1['data'][:, 0] += d1x
        self.s1['data'][:, 1] += d1y
        self.s2['data'][:, 0] += d2x
        self.s2['data'][:, 1] += d2y
        if self.s2['type'] == 'FR1' or self.s2['type'] == 'FR2':
            if self.s1['type'] == 'FR1' or self.s1['type'] == 'FR2':
                self.s2['core_x'] = self.s1['core_x']
                self.s2['core_y'] = self.s1['core_y']
            else:
                self.s2['core_x'] = self.s1['x']
                self.s2['core_y'] = self.s1['y']
        else:
            if self.s1['type'] == 'FR1' or self.s1['type'] == 'FR2':
                self.s2['x'] = self.s1['core_x']
                self.s2['y'] = self.s1['core_y']
            else:
                self.s2['x'] = self.s1['x']
                self.s2['y'] = self.s1['y']
        template_conv_beam2 =  self.template1_conv_beam2 + self.template2_conv_beam2
        self.s1['relative_x'] = int(dest_img_w / 2)
        self.s1['relative_y'] = int(dest_img_h / 2)
        self.s2['relative_x'] = int(dest_img_w / 2)
        self.s2['relative_y'] = int(dest_img_h / 2)
        
        return template_conv_beam2