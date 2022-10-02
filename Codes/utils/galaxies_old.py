# Copyright (c) 2021,2024 Yongkai Zhu <yongkai_zhu@hotmail.com>
# MIT License

import numpy as np

import matplotlib.pyplot as plt

from utils.spectrum import *
from utils.methods import *

class RSgen:
    def __init__(self, params):
        #image paramaters
        self.pix_area = params["pix_area_arcsec2"]
        self.pix_deg = params["pix_size_deg"]
        self.img_size = params["img_size"]
        self.pix_size = params["pix_size_arcsec"]
        self.fov_deg = params["fov_deg"]
        self.ra_min = -1 * self.fov_deg / 2.0
        self.dec_min = -1 * self.fov_deg / 2.0

    def rq(self, source, freq):
        flux_i_151 = 10 ** float(source['i_151'])
        ra = source['ra'] -  self.ra_min
        dec = source['dec'] - self.dec_min
        redshift = float(source['redshift'])
        galaxy = int(source["galaxy"])

        self.source = {
                "galaxy": galaxy,
                "type": "RQ",
                "pix": [],
                "freq": freq,
                "redshift": redshift,
                "flux_i_151": flux_i_151,
                "lobe1_i_151": 0,
                "lobe2_i_151": 0,
                "hotspot1_i_151": 0,
                "hotspot2_i_151": 0
            }
        x = np.round(ra / self.pix_deg)
        y = np.round(dec / self.pix_deg)
        x, y = int(x), int(y)
        flux = rqq_spec(flux_i_151, freq)
        Tb = calc_Tb(flux, self.pix_area, freq)
        image = []
        image.append([x, y, Tb, 0])
        self.source['pix'] = np.array(image)
                                  
        return self.source
        
    def sfsb(self, source, freq):
        m_hi = float(source['m_hi'])
        ra = float(source['ra']) - self.ra_min
        dec = float(source['dec']) - self.dec_min
        flux_i_151 = 10 ** float(source['i_151'])
        major_axis = float(source['major_axis'])
        minor_axis = float(source['minor_axis'])
        pa = float(source['pa'])
        redshift = float(source['redshift'])
        galaxy = int(source["galaxy"])
        sftype = int(source['sftype'])
        if sftype == 1:
            gtype = "SF"
        else:
            gtype = "SB"
            
        self.source = {
                "galaxy": galaxy,
                "type": gtype,
                "pix": [],
                "freq": freq,
                "redshift": redshift,
                "flux_i_151": flux_i_151,
                "lobe1_i_151": 0,
                "lobe2_i_151": 0,
                "hotspot1_i_151": 0,
                "hotspot2_i_151": 0
        }
        
        x = np.round(ra / self.pix_deg)
        y = np.round(dec / self.pix_deg)

        a = 0.5 * major_axis / self.pix_size
        b = 0.5 * minor_axis / self.pix_size
        c = np.sqrt(a * a - b * b)
        s = a * b * np.pi * self.pix_area
        if s < self.pix_area:
            s = self.pix_area
        f1x = x + c * np.sin(pa)
        f1y = y + c * np.cos(pa)
        f2x = x - c * np.sin(pa)
        f2y = y - c * np.cos(pa)  
        xmin = int(np.round(x - a))
        xmax = int(np.round(x + a))
        ymin = int(np.round(y - a))
        ymax = int(np.round(y + a))
        image = []
        print(xmin, xmax, ymin, ymax)
        for i in np.arange(xmin, xmax+1):
            for j in np.arange(ymin, ymax+1):
                if in_ellipse(i, j, f1x, f1y, f2x, f2y, a):
                    if sftype == 1:
                        Tb = calc_Tb(sf_spec(flux_i_151, freq), s, freq)
                        image.append([i, j, Tb, 0])
                    elif sftype == 2:
                        Tb = calc_Tb(sb_spec(flux_i_151, freq), s, freq)
                        image.append([i, j, Tb, 0])
        self.source['pix'] = np.array(image)                  
        return self.source
    
    def fr1(self, source, freq):
        #core
        core_index = source.index[0]
        ref_ra = self.ra_min
        ref_dec = self.dec_min
        core_flux_i_151 = 10 ** float(source.loc[core_index]['i_151'])
        core_ra = float(source.loc[core_index]['ra']) - ref_ra
        core_dec = float(source.loc[core_index]['dec']) - ref_dec
        redshift = float(source.loc[core_index]['redshift'])
        
        #lobe1
        lobe1_index = source.index[1]
        lobe1_flux_i_151 = 10 ** float(source.loc[lobe1_index]['i_151'])
        lobe1_ra = float(source.loc[lobe1_index]['ra']) - ref_ra
        lobe1_dec = float(source.loc[lobe1_index]['dec']) - ref_dec
        lobe1_pa = float(source.loc[lobe1_index]['pa'])
        lobe1_major_axis = float(source.loc[lobe1_index]['major_axis'])
        lobe1_minor_axis = float(source.loc[lobe1_index]['minor_axis'])
                                 
        #lobe2
        lobe2_index = source.index[2]
        lobe2_flux_i_151 = 10 ** float(source.loc[lobe2_index]['i_151'])
        lobe2_ra = float(source.loc[lobe2_index]['ra']) - ref_ra
        lobe2_dec = float(source.loc[lobe2_index]['dec']) - ref_dec
        lobe2_pa = float(source.loc[lobe2_index]['pa'])
        lobe2_major_axis = float(source.loc[lobe2_index]['major_axis'])
        lobe2_minor_axis = float(source.loc[lobe2_index]['minor_axis'])
        
        galaxy = int(source.loc[core_index]["galaxy"])
        gtype = "FR1"
    
        x = np.round(core_ra / self.pix_deg)
        y = np.round(core_dec / self.pix_deg)
        #lobe1
        x_lobe1 = lobe1_ra / self.pix_deg
        y_lobe1 = lobe1_dec / self.pix_deg
        #lobe2
        x_lobe2 = lobe2_ra / self.pix_deg
        y_lobe2 = lobe2_dec / self.pix_deg
        
        #core
        x, y = int(x), int(y)
        Tb = calc_Tb(fr1_core_spec(core_flux_i_151, freq), self.pix_area, freq)
        self.source = {
                "galaxy": galaxy,
                "type": gtype,
                "pix": [],
                "freq": freq,
                "redshift": redshift,
                "flux_i_151": core_flux_i_151,
                "lobe1_i_151": lobe1_flux_i_151,
                "lobe2_i_151": lobe2_flux_i_151,
                "hotspot1_i_151": 0,
                "hotspot2_i_151": 0,
            }             
        image = []  
        image.append([x, y, Tb, 0])
        
        #lobe1
        a1 = 0.5 * lobe1_major_axis / self.pix_size
        b1 = 0.5 * lobe1_minor_axis / self.pix_size
        c1 = np.sqrt(a1 * a1 - b1 * b1)
        s_lobe1 = a1 * b1 * np.pi * self.pix_area
        if s_lobe1 < self.pix_area:
            s_lobe1 = self.pix_area
        f11x = x_lobe1 + c1 * np.sin(lobe1_pa)
        f11y = y_lobe1 + c1 * np.cos(lobe1_pa)
        f21x = x_lobe1 - c1 * np.sin(lobe1_pa)
        f21y = y_lobe1 - c1 * np.cos(lobe1_pa)
            
        lobe1_xmin = int(np.round(x_lobe1 - a1))
        lobe1_xmax = int(np.round(x_lobe1 + a1))
        lobe1_ymin = int(np.round(y_lobe1 - a1))
        lobe1_ymax = int(np.round(y_lobe1 + a1))

        for i in np.arange(lobe1_xmin, lobe1_xmax+1):
            for j in np.arange(lobe1_ymin, lobe1_ymax+1):
                if in_ellipse(i, j, f11x, f11y, f21x, f21y, a1):
                    Tb = calc_Tb(fr1_lobe_spec(lobe1_flux_i_151, freq), s_lobe1, freq)
                    image.append([i, j, Tb, 1])
        #lobe2
        a2 = 0.5 * lobe2_major_axis / self.pix_size
        b2 = 0.5 * lobe2_minor_axis / self.pix_size
        c2 = np.sqrt(a2 * a2 - b2 * b2)
        s_lobe2 = a2 * b2 * np.pi * self.pix_area
        if s_lobe2 < self.pix_area:
            s_lobe2 = self.pix_area
        f12x = x_lobe2 + c2 * np.sin(lobe2_pa)
        f12y = y_lobe2 + c2 * np.cos(lobe2_pa)
        f22x = x_lobe2 - c2 * np.sin(lobe2_pa)
        f22y = y_lobe2 - c2 * np.cos(lobe2_pa)
            
        lobe2_xmin = int(np.round(x_lobe2 - a2))
        lobe2_xmax = int(np.round(x_lobe2 + a2))
        lobe2_ymin = int(np.round(y_lobe2 - a2))
        lobe2_ymax = int(np.round(y_lobe2 + a2))

        for i in np.arange(lobe2_xmin, lobe2_xmax+1):
            for j in np.arange(lobe2_ymin, lobe2_ymax+1):
                if in_ellipse(i, j, f12x, f12y, f22x, f22y, a2):
                    Tb = calc_Tb(fr1_lobe_spec(lobe2_flux_i_151, freq), s_lobe2, freq)
                    image.append([i, j, Tb, 1])
        self.source['pix'] = np.array(image) 
        return self.source['pix']
    
    def fr2(self, source, freq):
        #core
        core_index = source.index[0]
        ref_ra = self.ra_min
        ref_dec = self.dec_min
        core_flux_i_151 = 10 ** float(source.loc[core_index]['i_151'])
        core_ra = float(source.loc[core_index]['ra']) - ref_ra
        core_dec = float(source.loc[core_index]['dec']) - ref_dec
        redshift = float(source.loc[core_index]['redshift'])
        
        #lobe1
        lobe1_index = source.index[1]
        lobe1_flux_i_151 = 10 ** float(source.loc[lobe1_index]['i_151'])
        lobe1_ra = float(source.loc[lobe1_index]['ra']) - ref_ra
        lobe1_dec = float(source.loc[lobe1_index]['dec']) - ref_dec
        lobe1_pa = float(source.loc[lobe1_index]['pa'])
        lobe1_major_axis = float(source.loc[lobe1_index]['major_axis'])
        lobe1_minor_axis = float(source.loc[lobe1_index]['minor_axis'])
                                 
        #lobe2
        lobe2_index = source.index[2]
        lobe2_flux_i_151 = 10 ** float(source.loc[lobe2_index]['i_151'])
        lobe2_ra = float(source.loc[lobe2_index]['ra']) - ref_ra
        lobe2_dec = float(source.loc[lobe2_index]['dec']) - ref_dec
        lobe2_pa = float(source.loc[lobe2_index]['pa'])
        lobe2_major_axis = float(source.loc[lobe2_index]['major_axis'])
        lobe2_minor_axis = float(source.loc[lobe2_index]['minor_axis'])
        
        #hotspot1
        hotspot1_index = source.index[3]
        hotspot1_ra = float(source.loc[hotspot1_index]['ra']) - ref_ra
        hotspot1_dec = float(source.loc[hotspot1_index]['dec']) - ref_dec
        hotspot1_flux_i_151 = 10 ** float(source.loc[hotspot1_index]['i_151'])
        #hotspot2
        hotspot2_index = source.index[4]
        hotspot2_ra = float(source.loc[hotspot2_index]['ra']) - ref_ra
        hotspot2_dec = float(source.loc[hotspot2_index]['dec']) - ref_dec
        hotspot2_flux_i_151 = 10 ** float(source.loc[hotspot2_index]['i_151'])
        
        galaxy = int(source.loc[core_index]["galaxy"])
        gtype = "FR2"
    
        x = np.round(core_ra / self.pix_deg)
        y = np.round(core_dec / self.pix_deg)
        #lobe1
        x_lobe1 = lobe1_ra / self.pix_deg
        y_lobe1 = lobe1_dec / self.pix_deg
        #lobe2
        x_lobe2 = lobe2_ra / self.pix_deg
        y_lobe2 = lobe2_dec / self.pix_deg
        #hotspot1
        x_h1 = hotspot1_ra / self.pix_deg
        y_h1 = hotspot1_dec / self.pix_deg
        #hotspot2
        x_h2 = hotspot2_ra / self.pix_deg
        y_h2 = hotspot2_dec / self.pix_deg
        
        self.source = {
                "galaxy": galaxy,
                "type": gtype,
                "pix": [],
                "freq": freq,
                "redshift": redshift,
                "flux_i_151": core_flux_i_151,
                "lobe1_i_151": lobe1_flux_i_151,
                "lobe2_i_151": lobe2_flux_i_151,
                "hotspot1_i_151": hotspot1_flux_i_151,
                "hotspot2_i_151": hotspot2_flux_i_151,
            }   
        image = [] 
        #core
        x, y = int(x), int(y)
        Tb = calc_Tb(fr2_core_spec(core_flux_i_151, freq), self.pix_area, freq) 
        image.append([x, y, Tb, 0])
        
        #hotspot1
        x, y = int(x_h1), int(y_h1)
        Tb = calc_Tb(fr2_hotspot_spec(hotspot1_flux_i_151, freq), self.pix_area, freq)        
        image.append([x, y, Tb, 3])
        
        #hotspot2
        x, y = int(x_h2), int(y_h2)
        Tb = calc_Tb(fr2_hotspot_spec(hotspot2_flux_i_151, freq), self.pix_area, freq)        
        image.append([x, y, Tb, 4])
        
        #lobe1
        a1 = 0.5 * lobe1_major_axis / self.pix_size
        b1 = 0.5 * lobe1_minor_axis / self.pix_size
        c1 = np.sqrt(a1 * a1 - b1 * b1)
        s_lobe1 = a1 * b1 * np.pi * self.pix_area
        if s_lobe1 < self.pix_area:
            s_lobe1 = self.pix_area
        f11x = x_lobe1 + c1 * np.sin(lobe1_pa)
        f11y = y_lobe1 + c1 * np.cos(lobe1_pa)
        f21x = x_lobe1 - c1 * np.sin(lobe1_pa)
        f21y = y_lobe1 - c1 * np.cos(lobe1_pa)
            
        lobe1_xmin = int(np.round(x_lobe1 - a1))
        lobe1_xmax = int(np.round(x_lobe1 + a1))
        lobe1_ymin = int(np.round(y_lobe1 - a1))
        lobe1_ymax = int(np.round(y_lobe1 + a1))

        for i in np.arange(lobe1_xmin, lobe1_xmax+1):
            for j in np.arange(lobe1_ymin, lobe1_ymax+1):
                if in_ellipse(i, j, f11x, f11y, f21x, f21y, a1):
                    Tb = calc_Tb(fr2_lobe_spec(lobe1_flux_i_151, freq), s_lobe1, freq)
                    image.append([i, j, Tb, 1])
        #lobe2
        a2 = 0.5 * lobe2_major_axis / self.pix_size
        b2 = 0.5 * lobe2_minor_axis / self.pix_size
        c2 = np.sqrt(a2 * a2 - b2 * b2)
        s_lobe2 = a2 * b2 * np.pi * self.pix_area
        if s_lobe2 < self.pix_area:
            s_lobe2 = self.pix_area
        f12x = x_lobe2 + c2 * np.sin(lobe2_pa)
        f12y = y_lobe2 + c2 * np.cos(lobe2_pa)
        f22x = x_lobe2 - c2 * np.sin(lobe2_pa)
        f22y = y_lobe2 - c2 * np.cos(lobe2_pa)
            
        lobe2_xmin = int(np.round(x_lobe2 - a2))
        lobe2_xmax = int(np.round(x_lobe2 + a2))
        lobe2_ymin = int(np.round(y_lobe2 - a2))
        lobe2_ymax = int(np.round(y_lobe2 + a2))

        for i in np.arange(lobe2_xmin, lobe2_xmax+1):
            for j in np.arange(lobe2_ymin, lobe2_ymax+1):
                if in_ellipse(i, j, f12x, f12y, f22x, f22y, a2):
                    Tb = calc_Tb(fr2_lobe_spec(lobe2_flux_i_151, freq), s_lobe2, freq)
                    image.append([i, j, Tb, 1])
        self.source['pix'] = np.array(image) 
        return self.source['pix']
        
    
    def plot(self):
        new_x = self.source['pix'][:, 0] - np.min(self.source['pix'][:, 0])
        new_y = self.source['pix'][:, 1] - np.min(self.source['pix'][:, 1])
        h, w = [int(np.max(new_x))+1 , int(np.max(new_y))+1]
        print(h, w)
        img = np.zeros([h, w])
        for i in range(self.source['pix'].shape[0]):
            img[int(new_x[i]), int(new_y[i])] += self.source['pix'][i, 2]
        return img