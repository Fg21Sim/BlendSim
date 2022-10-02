# Copyright (c) 2021,2024 Yongkai Zhu <yongkai_zhu@hotmail.com>
# MIT License

import numpy as np
import pandas as pd

import utils.methods as mthd
from utils.methods import boundary_conditions

from astropy.io import fits

"""
Return pixel information of the simulated source.
x, y: x coordinate, y coordinate in a image coordinate system.
i_151: Flux of the pixel at position (x, y) at 151 MHz.
151: Simulation frequency
redshift: Redshift of the galaxy
int(agntype)/int(sftype): Galaxy type
    1: Radio quiet
    2: FR1
    3: FR2
    4: Star forming
    5: Star burst
int(galaxy): Galaxy number
i_151_tot: Total flux of the galaxy
i_151_core: Core flux
i_151_lobe1: Lobe flux
i_151_lobe2: Lobe flux
i_151_hotspot1: Hotspot flux
i_151_hotspot2: Hotspot flux
structure: Structure
"""

class RSgen:
    def __init__(self, PARAMS):
        """
        PARAMS (Dict): Simulation paramaters.
        """
        self.PARAMS = PARAMS
        
    def sfsb(self, source):
        """
        Star-formation / starburst galaxies

        """  
        gindex = source.index[0]
        galaxy = source.loc[gindex, 'galaxy'] # galaxy number
        sftype = source.loc[gindex, 'sftype'] + 3 # galaxy type; 
        i_151_tot = source.loc[gindex, 'i_151_tot']
        redshift = source.loc[gindex, 'redshift']
        structure = source.loc[gindex, 'structure']
        ra = source.loc[gindex, 'ra'] - self.PARAMS['ra_min']
        dec = source.loc[gindex, 'dec'] - self.PARAMS['dec_min']
        i_151 = source.loc[gindex, 'i_151_flux']
        pa = source.loc[gindex, 'pa']
        major = source.loc[gindex, 'major_axis']
        minor = source.loc[gindex, 'minor_axis']
        pix_info = []
        x = int(ra / self.PARAMS['pix_deg'])
        y = int(dec / self.PARAMS['pix_deg'])
        print(ra, dec, self.PARAMS['ra_min'])

        if boundary_conditions(0, self.PARAMS['img_size']):
            a = 0.5 * major / self.PARAMS['pix_size']
            b = 0.5 * minor / self.PARAMS['pix_size']
            xmin = int(np.round(x - a))
            xmax = int(np.round(x + a))
            ymin = int(np.round(y - a))
            ymax = int(np.round(y + a))
            xc = x - xmin
            yc = y - ymin
            ellipse = mthd.draw_ellipse([2*xc, 2*yc], [xc,yc], a, b, pa)
            area = np.sum(ellipse)
            if area < 1:
                area = 1
                pix = (x, y, i_151, 151, 
                       redshift, int(sftype), int(galaxy), 
                       i_151_tot, 
                       0, 0, 0, 0, 0,
                       structure)
                pix_info.append(pix)
                return pix_info
            else:
                exx, eyy = np.meshgrid(range(ellipse.shape[0]), range(ellipse.shape[1]))
                exx = exx.flatten()
                eyy = eyy.flatten()
                flux_pix = i_151 / area
                for i, j in zip(exx, eyy):
                    ii = i + xmin
                    jj = j + ymin
                    if ellipse[i, j] == 0:
                        continue
                    if boundary_conditions(0, self.PARAMS['img_size'], ii, jj):
                        if sftype == 4:
                            pix = (ii, jj, flux_pix, 151, 
                                   redshift, int(sftype), int(galaxy), 
                                   i_151_tot, 
                                   0, 0, 0, 0, 0,
                                   structure)
                            pix_info.append(pix)
                        elif sftype == 5:
                            pix = (ii, jj, flux_pix, 151, 
                                   redshift, int(sftype), int(galaxy), 
                                   i_151_tot, 
                                   0, 0, 0, 0, 0, 
                                   structure)               
                            pix_info.append(pix)
        return pix_info

    def rq(self, source):
        """
        Radio Quiet galaxies.

        """  
        gindex = source.index[0]
        i_151 = source.loc[gindex, 'i_151_flux']
        ra = source.loc[gindex, 'ra'] - self.PARAMS['ra_min']
        dec = source.loc[gindex, 'dec'] - self.PARAMS['dec_min']
        redshift = source.loc[gindex, 'redshift']
        galaxy = source.loc[gindex, 'galaxy']
        agntype = source.loc[gindex, 'agntype']
        structure = source.loc[gindex, 'structure']
        i_151_tot = source.loc[gindex, 'i_151_tot']
        
        x = int(ra / self.PARAMS['pix_deg'])
        y = int(dec / self.PARAMS['pix_deg'])
        pix_info = []
        if boundary_conditions(0, self.PARAMS['img_size'], x, y):
            pix = (x, y, i_151, 151, 
                   redshift, int(agntype), int(galaxy), 
                   i_151_tot, 
                   0, 0, 0, 0, 0, 
                   structure)
            pix_info.append(pix)
        return pix_info
    
    def fr1(self, source, fr1_temp_data, fr1_temp_hdr):
        """
        Model FRI radio galaxies with templates.
        """
        index = source.index
        galaxy = source.loc[index[0]].galaxy
        agntype = source.loc[index[0]].agntype
        i_151_tot = source.loc[index[0]].i_151_tot
        redshift = source.loc[index[0]].redshift
        #core
        core_structure = source.loc[index[0]].structure
        core_ra = source.loc[index[0]].ra - self.PARAMS['ra_min']
        core_dec = source.loc[index[0]].dec - self.PARAMS['dec_min']
        core_i_151 = source.loc[index[0]].i_151_flux
        #lobel
        lobe1_structure = source.loc[index[1]].structure
        lobe1_ra = source.loc[index[1]].ra - self.PARAMS['ra_min']
        lobe1_dec = source.loc[index[1]].dec - self.PARAMS['dec_min']
        lobe1_i_151 = source.loc[index[1]].i_151_flux
        lobe1_pa = source.loc[index[1]].pa
        lobe1_major = source.loc[index[1], 'major_axis']
        lobe1_minor = source.loc[index[1], 'minor_axis']
        #lobe2
        lobe2_structure = source.loc[index[2]].structure
        lobe2_ra = source.loc[index[2]].ra - self.PARAMS['ra_min']
        lobe2_dec = source.loc[index[2]].dec - self.PARAMS['dec_min']
        lobe2_i_151 = source.loc[index[2]].i_151_flux
        lobe2_pa = source.loc[index[2]].pa
        lobe2_major = source.loc[index[2], 'major_axis']
        lobe2_minor = source.loc[index[2], 'minor_axis']

        core_x, lobe1_x, lobe2_x = (source.ra - self.PARAMS['ra_min']) / self.PARAMS['pix_deg']
        core_y, lobe1_y, lobe2_y = (source.dec - self.PARAMS['dec_min']) / self.PARAMS['pix_deg']

        if not boundary_conditions(0, self.PARAMS['img_size'], 
                                   core_x, lobe1_x, lobe2_x, 
                                   core_y, lobe1_y, lobe2_y):
            pix_info = None
            return pix_info

        major_axis1 = lobe1_major / self.PARAMS['pix_size']
        major_axis2 = lobe2_major / self.PARAMS['pix_size']
        distance_lobe1_core = np.sqrt((lobe1_x - core_x) ** 2 + (lobe1_y - core_y) ** 2)
        source_size = distance_lobe1_core * 2 + major_axis1

        temp_xc = int(fr1_temp_data.shape[0] / 2)
        temp_yc = int(fr1_temp_data.shape[1] / 2)
        scaling_factor = source_size / fr1_temp_hdr['MAJOR_AX']
        rotating_angle = -1 * fr1_temp_hdr['ANGLE'] - lobe1_pa
        rotating_angle = rotating_angle * 180 / np.pi
        rot_img = mthd.rotate_image(fr1_temp_data, rotating_angle)
        try:
            rescaled_image = mthd.rescale_image(rot_img, scaling_factor)
        except:
            pix_info = self.fr1Wilman(source)
            return pix_info
        else:
            fr1_area = np.sum(rescaled_image[rescaled_image != 0]) * self.PARAMS['pix_area']
            if np.sum(rescaled_image) == 0:
                pix_info = self.fr1Wilman(source)
                return pix_info
            else:
                fr1_norm = rescaled_image / np.sum(rescaled_image)
                fr1_norm_index = np.argwhere(fr1_norm > 0)
                fr1_norm_p = fr1_norm[fr1_norm_index[:, 0], fr1_norm_index[:, 1]]
                core_x, core_y = int(core_x), int(core_y)
                fr1_xc = int( fr1_norm.shape[0] / 2 )
                fr1_yc = int( fr1_norm.shape[1] / 2 )
                images_tmp = i_151_tot * fr1_norm_p
                new_x = fr1_norm_index[:, 0] - fr1_xc + core_x
                new_y = fr1_norm_index[:, 1] - fr1_yc + core_y
                if not boundary_conditions(0, self.PARAMS['img_size'], new_x) \
                    or not boundary_conditions(0, self.PARAMS['img_size'], new_y):
                    pix_info = None
                    return pix_info
                pix_info = []
                for i in range(len(images_tmp)):
                    pix_info.append((new_x[i], new_y[i], images_tmp[i], 151,
                                     redshift, int(agntype), int(galaxy),
                                     i_151_tot,
                                     core_i_151, lobe1_i_151, lobe2_i_151, 0, 0,
                                     3))
        return pix_info


    def fr2(self, source, fr2_temp_data, fr2_temp_hdr):
        """
        Model FRII radio galaxies with templates.
        """
        index = source.index
        galaxy = source.loc[index[0]].galaxy
        agntype = source.loc[index[0]].agntype
        i_151_tot = source.loc[index[0]].i_151_tot
        redshift = source.loc[index[0]].redshift
        #core
        core_structure = source.loc[index[0]].structure
        core_ra = source.loc[index[0]].ra - self.PARAMS['ra_min']
        core_dec = source.loc[index[0]].dec - self.PARAMS['dec_min']
        core_i_151 = source.loc[index[0]].i_151_flux
        #lobel
        lobe1_structure = source.loc[index[1]].structure
        lobe1_ra = source.loc[index[1]].ra - self.PARAMS['ra_min']
        lobe1_dec = source.loc[index[1]].dec - self.PARAMS['dec_min']
        lobe1_i_151 = source.loc[index[1]].i_151_flux
        lobe1_pa = source.loc[index[1]].pa
        lobe1_major = source.loc[index[1], 'major_axis']
        lobe1_minor = source.loc[index[1], 'minor_axis']
        #lobe2
        lobe2_structure = source.loc[index[2]].structure
        lobe2_ra = source.loc[index[2]].ra - self.PARAMS['ra_min']
        lobe2_dec = source.loc[index[2]].dec - self.PARAMS['dec_min']
        lobe2_i_151 = source.loc[index[2]].i_151_flux
        lobe2_pa = source.loc[index[2]].pa
        lobe2_major = source.loc[index[2], 'major_axis']
        lobe2_minor = source.loc[index[2], 'minor_axis']
        #hotspot1
        hotspot1_ra = source.loc[index[3]].ra - self.PARAMS['ra_min']
        hotspot1_dec = source.loc[index[3]].dec - self.PARAMS['dec_min']
        hotspot1_i_151 = source.loc[index[3]].i_151_flux
        hotspot1_structure = source.loc[index[3]].structure
        #hotspot2
        hotspot2_ra = source.loc[index[4]].ra - self.PARAMS['ra_min']
        hotspot2_dec = source.loc[index[4]].dec - self.PARAMS['dec_min']
        hotspot2_i_151 = source.loc[index[4]].i_151_flux
        hotspot2_structure = source.loc[index[4]].structure    

        core_x, lobe1_x, lobe2_x, hp1_x, hp2_x = (source.ra - self.PARAMS['ra_min']) / self.PARAMS['pix_deg']
        core_y, lobe1_y, lobe2_y, hp1_y, hp2_y = (source.dec - self.PARAMS['dec_min']) / self.PARAMS['pix_deg']
        pix_info = []
        if not boundary_conditions(0, self.PARAMS['img_size'],
                               core_x, lobe1_x, lobe2_x, hp1_x, hp2_x,
                               core_y, lobe1_y, lobe2_y, hp1_y, hp2_y):
            pix_info = None
            return pix_info
        major_axis1 = lobe1_major / self.PARAMS['pix_size']
        major_axis2 = lobe2_major / self.PARAMS['pix_size']
        distance_lobe1_core = np.sqrt((lobe1_x - core_x) ** 2 + (lobe1_y - core_y) ** 2)
        source_size = distance_lobe1_core * 2 + major_axis1
        temp_xc = int(fr2_temp_data.shape[0] / 2)
        temp_yc = int(fr2_temp_data.shape[1] / 2)
        scaling_factor = source_size / fr2_temp_hdr['MAJOR_AX']
        rotating_angle = -1 * fr2_temp_hdr['ANGLE'] - lobe1_pa
        rotating_angle = rotating_angle * 180 / np.pi
        rot_img = mthd.rotate_image(fr2_temp_data, rotating_angle)
        try:
            rescaled_image = mthd.rescale_image(rot_img, scaling_factor)
        except:
            pix_info = self.fr2Wilman(source)
            return pix_info
        else:
            fr2_area = np.sum(rescaled_image[rescaled_image != 0]) * self.PARAMS['pix_area']
            if np.sum(rescaled_image) == 0:
                pix_info = self.fr2Wilman(source)
                return pix_info
            else:
                fr2_norm = rescaled_image / np.sum(rescaled_image)
                fr2_norm_index = np.argwhere(fr2_norm > 0)
                fr2_norm_p = fr2_norm[fr2_norm_index[:, 0], fr2_norm_index[:, 1]]
                core_x, core_y = int(core_x), int(core_y)
                fr2_xc = int( fr2_norm.shape[0] / 2 )
                fr2_yc = int( fr2_norm.shape[1] / 2 )
                images_tmp = i_151_tot * fr2_norm_p
                new_x = fr2_norm_index[:, 0] - fr2_xc + core_x
                new_y = fr2_norm_index[:, 1] - fr2_yc + core_y
                if not boundary_conditions(0, self.PARAMS['img_size'], new_x) \
                    or not boundary_conditions(0, self.PARAMS['img_size'], new_y):
                    pix_info = None
                    return pix_info
                pix_info = []
                for i in range(len(images_tmp)):
                    pix_info.append((new_x[i], new_y[i], images_tmp[i], 151,
                                     redshift, int(agntype), int(galaxy),
                                     i_151_tot,
                                     core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151,
                                     3))
        return pix_info

    def fr1Wilman(self, source):
        """
        Model FRI radio galaxies using Wilman's model.
        """
        index = source.index
        galaxy = source.loc[index[0]].galaxy
        agntype = source.loc[index[0]].agntype
        i_151_tot = source.loc[index[0]].i_151_tot
        redshift = source.loc[index[0]].redshift
        #core
        core_structure = source.loc[index[0]].structure
        core_ra = source.loc[index[0]].ra - self.PARAMS['ra_min']
        core_dec = source.loc[index[0]].dec - self.PARAMS['dec_min']
        core_i_151 = source.loc[index[0]].i_151_flux
        #lobel
        lobe1_structure = source.loc[index[1]].structure
        lobe1_ra = source.loc[index[1]].ra - self.PARAMS['ra_min']
        lobe1_dec = source.loc[index[1]].dec - self.PARAMS['dec_min']
        lobe1_i_151 = source.loc[index[1]].i_151_flux
        lobe1_pa = source.loc[index[1]].pa
        lobe1_major = source.loc[index[1], 'major_axis']
        lobe1_minor = source.loc[index[1], 'minor_axis']
        #lobe2
        lobe2_structure = source.loc[index[2]].structure
        lobe2_ra = source.loc[index[2]].ra - self.PARAMS['ra_min']
        lobe2_dec = source.loc[index[2]].dec - self.PARAMS['dec_min']
        lobe2_i_151 = source.loc[index[2]].i_151_flux
        lobe2_pa = source.loc[index[2]].pa
        lobe2_major = source.loc[index[2], 'major_axis']
        lobe2_minor = source.loc[index[2], 'minor_axis']

        core_x, lobe1_x, lobe2_x = (source.ra - self.PARAMS['ra_min']) / self.PARAMS['pix_deg']
        core_y, lobe1_y, lobe2_y = (source.dec - self.PARAMS['dec_min']) / self.PARAMS['pix_deg']
        pix_info = []
        if not boundary_conditions(0, self.PARAMS['img_size'], 
                                   core_x, lobe1_x, lobe2_x, 
                                   core_y, lobe1_y, lobe2_y):
            pix_info = None
            return pix_info
        #core
        pix_info.append((int(core_x), int(core_y), core_i_151, 151,
                         redshift, int(agntype), int(galaxy),
                         i_151_tot,
                         core_i_151, lobe1_i_151, lobe2_i_151, 0, 0, 
                         core_structure))
        #lobe1
        a1 = 0.5 * lobe1_major / self.PARAMS['pix_size']
        b1 = 0.5 * lobe1_minor / self.PARAMS['pix_size']
        lobe1_xmin = int(np.round(lobe1_x - a1))
        lobe1_xmax = int(np.round(lobe1_x + a1))
        lobe1_ymin = int(np.round(lobe1_y - a1))
        lobe1_ymax = int(np.round(lobe1_y + a1))

        lobe1_xc = int(lobe1_x) - lobe1_xmin
        lobe1_yc = int(lobe1_y) - lobe1_ymin
        pa = lobe1_pa
        ellipse1 = mthd.draw_ellipse([2*lobe1_xc, 2*lobe1_yc],
                                       [lobe1_xc,lobe1_yc],
                                       a1, b1, pa)
        area1 = np.sum(ellipse1)
        if area1 == 0:
            area1 = 1
            pix_info.append((int(lobe1_x), int(lobe1_y), lobe1_i_151, 151,
                             redshift, int(agntype), int(galaxy),
                             i_151_tot,
                             core_i_151, lobe1_i_151, lobe2_i_151, 0, 0,
                             lobe1_structure))
        else:
            exx, eyy = np.meshgrid(range(ellipse1.shape[0]), range(ellipse1.shape[1]))
            exx = exx.flatten()
            eyy = eyy.flatten()
            flux_pix = lobe1_i_151 / area1
            for i, j in zip(exx, eyy):
                ii = i + lobe1_xmin
                jj = j + lobe1_ymin
                if ellipse1[i, j] == 0:
                    continue
                if boundary_conditions(0, self.PARAMS['img_size'], ii, jj):
                    pix_info.append((ii, jj, flux_pix, 151,
                                     redshift, int(agntype), int(galaxy),
                                     i_151_tot,
                                     core_i_151, lobe1_i_151, lobe2_i_151, 0, 0,
                                     lobe1_structure))
        #lobe2
        a1 = 0.5 * lobe2_major / self.PARAMS['pix_size']
        b1 = 0.5 * lobe2_minor / self.PARAMS['pix_size']
        lobe2_xmin = int(np.round(lobe2_x - a1))
        lobe2_xmax = int(np.round(lobe2_x + a1))
        lobe2_ymin = int(np.round(lobe2_y - a1))
        lobe2_ymax = int(np.round(lobe2_y + a1))

        lobe2_xc = int(lobe2_x) - lobe2_xmin
        lobe2_yc = int(lobe2_y) - lobe2_ymin
        pa = lobe2_pa
        ellipse1 = mthd.draw_ellipse([2*lobe2_xc, 2*lobe2_yc],
                                       [lobe2_xc,lobe2_yc],
                                       a1, b1, pa)
        area1 = np.sum(ellipse1)
        if area1 == 0:
            area1 = 1
            pix_info.append((int(lobe2_x), int(lobe2_y), lobe2_i_151, 151,
                             redshift, int(agntype), int(galaxy),
                             i_151_tot,
                             core_i_151, lobe1_i_151, lobe2_i_151, 0, 0,
                             lobe2_structure))
        else:
            exx, eyy = np.meshgrid(range(ellipse1.shape[0]), range(ellipse1.shape[1]))
            exx = exx.flatten()
            eyy = eyy.flatten()
            flux_pix = lobe2_i_151 / area1
            for i, j in zip(exx, eyy):
                ii = i + lobe2_xmin
                jj = j + lobe2_ymin
                if ellipse1[i, j] == 0:
                    continue
                if boundary_conditions(0, self.PARAMS['img_size'], ii, jj):
                    pix_info.append((ii, jj, flux_pix, 151,
                             redshift, int(agntype), int(galaxy),
                             i_151_tot,
                             core_i_151, lobe1_i_151, lobe2_i_151, 0, 0,
                             lobe2_structure))
        return pix_info


    def fr2Wilman(self, source):
        """
        Model FRII radio galaxies using Wilman's model.
        """
        index = source.index
        galaxy = source.loc[index[0]].galaxy
        agntype = source.loc[index[0]].agntype
        i_151_tot = source.loc[index[0]].i_151_tot
        redshift = source.loc[index[0]].redshift
        #core
        core_structure = source.loc[index[0]].structure
        core_ra = source.loc[index[0]].ra - self.PARAMS['ra_min']
        core_dec = source.loc[index[0]].dec - self.PARAMS['dec_min']
        core_i_151 = source.loc[index[0]].i_151_flux
        #lobel
        lobe1_structure = source.loc[index[1]].structure
        lobe1_ra = source.loc[index[1]].ra - self.PARAMS['ra_min']
        lobe1_dec = source.loc[index[1]].dec - self.PARAMS['dec_min']
        lobe1_i_151 = source.loc[index[1]].i_151_flux
        lobe1_pa = source.loc[index[1]].pa
        lobe1_major = source.loc[index[1], 'major_axis']
        lobe1_minor = source.loc[index[1], 'minor_axis']
        #lobe2
        lobe2_structure = source.loc[index[2]].structure
        lobe2_ra = source.loc[index[2]].ra - self.PARAMS['ra_min']
        lobe2_dec = source.loc[index[2]].dec - self.PARAMS['dec_min']
        lobe2_i_151 = source.loc[index[2]].i_151_flux
        lobe2_pa = source.loc[index[2]].pa
        lobe2_major = source.loc[index[2], 'major_axis']
        lobe2_minor = source.loc[index[2], 'minor_axis']
        #hotspot1
        hotspot1_ra = source.loc[index[3]].ra - self.PARAMS['ra_min']
        hotspot1_dec = source.loc[index[3]].dec - self.PARAMS['dec_min']
        hotspot1_i_151 = source.loc[index[3]].i_151_flux
        hotspot1_structure = source.loc[index[3]].structure
        #hotspot2
        hotspot2_ra = source.loc[index[4]].ra - self.PARAMS['ra_min']
        hotspot2_dec = source.loc[index[4]].dec - self.PARAMS['dec_min']
        hotspot2_i_151 = source.loc[index[4]].i_151_flux
        hotspot2_structure = source.loc[index[4]].structure    

        core_x, lobe1_x, lobe2_x, hp1_x, hp2_x = (source.ra - self.PARAMS['ra_min']) / self.PARAMS['pix_deg']
        core_y, lobe1_y, lobe2_y, hp1_y, hp2_y = (source.dec - self.PARAMS['dec_min']) / self.PARAMS['pix_deg']
        pix_info = []
        if boundary_conditions(0, self.PARAMS['img_size'],
                               core_x, lobe1_x, lobe2_x, hp1_x, hp2_x,
                               core_y, lobe1_y, lobe2_y, hp1_y, hp2_y
                              ):
            core_x, core_y, hp1_x, hp1_y, hp2_x, hp2_y = list(map(int, 
                                [core_x, core_y, hp1_x, hp1_y, hp2_x, hp2_y]))
            #core
            pix = (core_x, core_y, core_i_151, 151, 
                   redshift, int(agntype), int(galaxy), 
                   i_151_tot,
                   core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151,
                   core_structure)
            pix_info.append(pix)
            #hotspot1
            pix = (hp1_x, hp1_y, hotspot1_i_151, 151, 
                   redshift, int(agntype), int(galaxy), 
                   i_151_tot,
                   core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151,
                   hotspot1_structure)
            pix_info.append(pix)
            #hotspot2
            pix = (hp2_x, hp2_y, hotspot2_i_151, 151, 
                   redshift, int(agntype), int(galaxy), 
                   i_151_tot,
                   core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151, 
                   hotspot2_structure)
            pix_info.append(pix)

            #lobe1
            #lobe1_x = lobe1_ra / self.PARAMS['pix_size']
            #lobe1_y = lobe1_dec / self.PARAMS['pix_size']
            a1 = 0.5 * lobe1_major / self.PARAMS['pix_size']
            b1 = 0.5 * lobe1_minor / self.PARAMS['pix_size']

            lobe1_xmin = int(np.round(lobe1_x - a1))
            lobe1_xmax = int(np.round(lobe1_x + a1))
            lobe1_ymin = int(np.round(lobe1_y - a1))
            lobe1_ymax = int(np.round(lobe1_y + a1))

            lobe1_xc = int(lobe1_x) - lobe1_xmin
            lobe1_yc = int(lobe1_y) - lobe1_ymin
            pa = lobe1_pa
            ellipse = mthd.draw_ellipse([2*lobe1_xc, 2*lobe1_yc],
                                       [lobe1_xc,lobe1_yc],
                                       a1, b1, pa)
            area = np.sum(ellipse)
            if area == 0:
                area = 1
                pix = (hp1_x, hp1_y, lobe1_i_151, 151, 
                       redshift, int(agntype), int(galaxy), 
                       i_151_tot,
                       core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151, 
                       lobe1_structure)
                pix_info.append(pix)
            else:
                exx, eyy = np.meshgrid(range(ellipse.shape[0]), range(ellipse.shape[1]))
                exx = exx.flatten()
                eyy = eyy.flatten()
                flux_pix = lobe1_i_151 / area
                for i, j in zip(exx, eyy):
                    ii = i + lobe1_xmin
                    jj = j + lobe1_ymin
                    if ellipse[i, j] == 0:
                        continue
                    if boundary_conditions(0, self.PARAMS['img_size'], ii, jj):
                        pix = (ii, jj, flux_pix, 151, 
                               redshift, int(agntype), int(galaxy), 
                               i_151_tot,
                               core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151, 
                               lobe1_structure)
                        pix_info.append(pix)

            #lobe2
            #lobe2_x = lobe2_ra / self.PARAMS['pix_size']
            #lobe2_y = lobe2_dec / self.PARAMS['pix_size']
            a1 = 0.5 * lobe2_major / self.PARAMS['pix_size']
            b1 = 0.5 * lobe2_minor / self.PARAMS['pix_size']

            lobe2_xmin = int(np.round(lobe2_x - a1))
            lobe2_xmax = int(np.round(lobe2_x + a1))
            lobe2_ymin = int(np.round(lobe2_y - a1))
            lobe2_ymax = int(np.round(lobe2_y + a1))

            lobe2_xc = int(lobe2_x) - lobe2_xmin
            lobe2_yc = int(lobe2_y) - lobe2_ymin
            pa = lobe2_pa
            ellipse = mthd.draw_ellipse([2*lobe2_xc, 2*lobe2_yc],
                                       [lobe2_xc,lobe2_yc],
                                       a1, b1, pa)
            area = np.sum(ellipse)
            if area == 0:
                area = 1
                pix = (hp1_x, hp1_y, lobe2_i_151, 151, 
                       redshift, int(agntype), int(galaxy), 
                       i_151_tot,
                       core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151,
                       lobe2_structure)
                pix_info.append(pix)
            else:
                exx, eyy = np.meshgrid(range(ellipse.shape[0]), range(ellipse.shape[1]))
                exx = exx.flatten()
                eyy = eyy.flatten()
                flux_pix = lobe2_i_151 / area
                for i, j in zip(exx, eyy):
                    ii = i + lobe2_xmin
                    jj = j + lobe2_ymin
                    if ellipse[i, j] == 0:
                        continue
                    if boundary_conditions(0, self.PARAMS['img_size'], ii, jj):
                        pix = (ii, jj, flux_pix, 151, 
                               redshift, int(agntype), int(galaxy), 
                               i_151_tot,
                               core_i_151, lobe1_i_151, lobe2_i_151, hotspot1_i_151, hotspot2_i_151,
                               lobe2_structure)
                        pix_info.append(pix)
        return pix_info
