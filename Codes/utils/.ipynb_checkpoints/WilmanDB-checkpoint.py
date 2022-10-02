# Copyright (c) 2021,2024 Yongkai Zhu <yongkai_zhu@hotmail.com>
# MIT License

import os
import random

import numpy as np
import pandas as pd

from utils.spectrum import *
from utils.methods import *

class WilmanDB:
    def __init__(self, dbpath = None):
        """
        Load data from Wilman data.
        """
        if dbpath == None:
            work_dir = os.getcwd()
            dbpath = os.path.join(work_dir, 'data')
        
        rqfn = dbpath + '/wilmandb_rqq.h5'
        fr1fn = dbpath + '/wilmandb_fr1.h5'
        fr2fn = dbpath + '/wilmandb_fr2.h5'
        sffn = dbpath + '/wilmandb_sf.h5'
        sbfn = dbpath + '/wilmandb_sb.h5'
        
        print("Loading data from %s." %(rqfn))
        self.rq = pd.read_hdf(rqfn, key='rqq')
        print("Finish loading data")
        print("Loading data from %s." %(fr1fn))
        self.fr1 = pd.read_hdf(fr1fn, key='fr1')
        print("Finish loading data")
        print("Loading data from %s." %(fr2fn))
        self.fr2 = pd.read_hdf(fr2fn, key='fr2')
        print("Finish loading data")
        print("Loading data from %s." %(sffn))
        self.sf = pd.read_hdf(sffn, key='sf')
        print("Finish loading data")
        print("Loading data from %s." %(sbfn))
        self.sb = pd.read_hdf(sbfn, key='sb')
        print("Finish loading data")
        
        self.rq['type'] = 'RQQ'
        self.fr1["type"] = "FR1"
        self.fr2["type"] = "FR2"
        self.sf["type"] = "SF"
        self.sb["type"] = "SB"
        
        self.rqdb = pd.read_csv(dbpath + "/wilman_db/wilman_db_rq_-2.5-2.5v2.csv")
        self.fr1db = pd.read_csv(dbpath + "/wilman_db/wilman_db_fr1-2.5-2.5v2.csv")
        self.fr2db = pd.read_csv(dbpath + "/wilman_db/wilman_db_fr2-2.5-2.5v2.csv")
        self.sfdb = pd.read_csv(dbpath + "/wilman_db/wilman_db_sf_-2.5-2.5v2.csv")
        self.sbdb = pd.read_csv(dbpath + "/wilman_db/wilman_db_sb_-2.5-2.5v2.csv")
        
   
    def sampling1(self, number_of_galaxies, freq, flux_min, flux_max, fov, bins):
        flux_freq = "i_" + str(freq)
        if flux_freq not in self.rqdb.keys():
            self.rqdb[flux_freq] = rqq_spec(self.rqdb["i_151"], freq)
        if flux_freq not in self.fr1.keys():
            self.fr1db[flux_freq] = fr1_core_spec( self.fr1db["core_i_151"], freq ) \
                + fr1_lobe_spec( self.fr1db["lobe1_i_151"], freq) \
                + fr1_lobe_spec( self.fr1db["lobe2_i_151"], freq)
        if flux_freq not in self.fr2.keys():
            self.fr2db[flux_freq] = fr2_core_spec( self.fr2db["core_i_151"], freq ) \
                + fr2_lobe_spec( self.fr2db["lobe1_i_151"], freq) \
                + fr2_lobe_spec( self.fr2db["lobe2_i_151"], freq)\
                + fr2_hotspot_spec( self.fr2db["hotspot1_i_151"], freq) \
                + fr2_hotspot_spec( self.fr2db["hotspot2_i_151"], freq)
        if flux_freq not in self.sf.keys():
            self.sfdb[flux_freq] = sf_spec(self.sfdb["i_151"], freq)
        if flux_freq not in self.sb.keys():
            self.sbdb[flux_freq] = sb_spec(self.sbdb["i_151"], freq)

        fovl, fovu = fov
        fr1 = self.fr1db[(self.fr1db["ra"]>fovl) & (self.fr1db["ra"]<fovu) \
                       & (self.fr1db["dec"]>fovl) & (self.fr1db["dec"]<fovu)]
        fr2 = self.fr2db[(self.fr2db["ra"]>fovl) & (self.fr2db["ra"]<fovu) \
                       & (self.fr2db["dec"]>fovl) & (self.fr2db["dec"]<fovu)]
        rq = self.rqdb[(self.rqdb["ra"]>fovl) & (self.rqdb["ra"]<fovu) \
                     & (self.rqdb["dec"]>fovl) & (self.rqdb["dec"]<fovu)]
        sf = self.sfdb[(self.sfdb["ra"]>fovl) & (self.sfdb["ra"]<fovu) \
                     & (self.sfdb["dec"]>fovl) & (self.sfdb["dec"]<fovu)]
        sb = self.sbdb[(self.sbdb["ra"]>fovl) & (self.sbdb["ra"]<fovu) \
                     & (self.sbdb["dec"]>fovl) & (self.sbdb["dec"]<fovu)]
        flux_min = np.max([1e-8, flux_min])
        flux_max = np.min([flux_max, np.max([np.max(rq[flux_freq]), \
                                             np.max(fr1[flux_freq]), \
                                             np.max(fr2[flux_freq]), \
                                             np.max(sf[flux_freq]), \
                                             np.max(sb[flux_freq])])])
        print("Minimum flux: %.2e; Maximum flux: %.2e." %(flux_min, flux_max))
        if number_of_galaxies == -1:
            galaxies = pd.concat([rq, fr1, fr2, sf, sb], ignore_index=True, sort=True)
            return galaxies
        
        logspace = np.logspace(np.log10(flux_min), np.log10(flux_max), bins)
        histfr1, bin_edgesfr1 = np.histogram(fr1[flux_freq], bins=logspace)
        histfr2, bin_edgesfr2 = np.histogram(fr2[flux_freq], bins=logspace)
        histrq, bin_edgesrq = np.histogram(rq[flux_freq], bins=logspace)
        histsf, bin_edgessf = np.histogram(sf[flux_freq], bins=logspace)
        histsb, bin_edgessb = np.histogram(sb[flux_freq], bins=logspace)
        bin_edges = bin_edgesfr1
        histtot = histfr1 + histfr2 + histrq + histsb + histsf
        histcd = np.zeros(bins + 1)
        for i in range(bins + 1):
            histcd[i] = np.sum(histtot[0:i])
        histcd = histcd / np.max(histcd)
        columns = ['ra', 'dec', flux_freq, "type", "galaxy"]
        galaxies = pd.DataFrame(columns=columns, dtype=object)
        while len(galaxies) < number_of_galaxies:
            randn = np.random.rand(1)[0]
            index = np.argwhere(histcd >= randn)[0][0]
            classhist = [histfr1[index-1: index], histfr2[index-1: index], \
                         histrq[index-1: index], histsf[index-1: index], histsb[index-1: index]]
            classhist = np.array(classhist)
            histcd2 = np.zeros(6)
            for i in range(6):
                histcd2[i] = np.sum(classhist[0:i])
            if np.sum(histcd2) == 0:
                continue
            histcd2 = histcd2 / np.max(histcd2)
            #if histcd2[1:].min() == histcd2[1:].max() or histcd2[1] == 0:
            #    continue
            #b = histcd2
            #b[1:]=(b[1:] - b[1:].min())/(b[1:].max() - b[1:].min()) * 0.8 + 0.2
            #histcd2 = b
            randn = np.random.rand(1)[0]
            index2 = np.argwhere(histcd2 >= randn)[0][0]
            if index2 == 1:
                agntype = "FR1"
                flux1 = bin_edges[index-1]
                flux2 = bin_edges[index]
                tmp = fr1[fr1[flux_freq] > flux1]
                tmp = tmp[tmp[flux_freq] <= flux2]
                
                if len(tmp) == 0:
                    continue
                index = np.random.choice(tmp.index)
                ggg = tmp.loc[index, "galaxy"]
                if ggg in list(galaxies["galaxy"]):
                    continue
                galaxies = galaxies.append({"ra": tmp.loc[index, "ra"],
                                        "dec": tmp.loc[index, "dec"],
                                        "i_151": tmp.loc[index, "i_151"],
                                        "core_i_151": tmp.loc[index, "core_i_151"],
                                        "lobe1_i_151": tmp.loc[index, "lobe1_i_151"],
                                        "lobe2_i_151": tmp.loc[index, "lobe2_i_151"],
                                        flux_freq: tmp.loc[index, flux_freq],
                                        "type": tmp.loc[index, "type"],
                                        "galaxy": tmp.loc[index, "galaxy"],
                
                }, ignore_index=True) 

            elif index2 == 2:
                agntype = "FR2"
                flux1 = bin_edges[index-1]
                flux2 = bin_edges[index]
                tmp = fr2[fr2[flux_freq] > flux1]
                tmp = tmp[tmp[flux_freq] <= flux2]
                if len(tmp) == 0:
                    continue
                index = np.random.choice(tmp.index)
                ggg = tmp.loc[index, "galaxy"]
                if ggg in list(galaxies["galaxy"]):
                    continue

                galaxies = galaxies.append({"ra": tmp.loc[index, "ra"],
                                        "dec": tmp.loc[index, "dec"],
                                        "i_151": tmp.loc[index, "i_151"],
                                        "core_i_151": tmp.loc[index, "core_i_151"],
                                        "lobe1_i_151": tmp.loc[index, "lobe1_i_151"],
                                        "lobe2_i_151": tmp.loc[index, "lobe2_i_151"],
                                        "hotspot1_i_151": tmp.loc[index, "hotspot1_i_151"],
                                        "hotspot2_i_151": tmp.loc[index, "hotspot2_i_151"],
                                        flux_freq: tmp.loc[index, flux_freq],
                                        "type": tmp.loc[index, "type"],
                                        "galaxy": tmp.loc[index, "galaxy"],
                
                }, ignore_index=True) 
            elif index2 == 3:
                agntype = "RQQ"
                flux1 = bin_edges[index-1]
                flux2 = bin_edges[index]
                tmp = rq[rq[flux_freq] > flux1]
                tmp = tmp[tmp[flux_freq] <= flux2]
                if len(tmp) == 0:
                    continue
                index = np.random.choice(tmp.index)
                ggg = tmp.loc[index, "galaxy"]
                if ggg in list(galaxies["galaxy"]):
                    continue
                galaxies = galaxies.append({"ra": tmp.loc[index, "ra"],
                                        "dec": tmp.loc[index, "dec"],
                                        "i_151": tmp.loc[index, "i_151"],
                                        flux_freq: tmp.loc[index, flux_freq],
                                        "type": tmp.loc[index, "type"],
                                        "galaxy": tmp.loc[index, "galaxy"],
                
                }, ignore_index=True) 

            elif index2 == 4:
                agntype = "SF"
                flux1 = bin_edges[index-1]
                flux2 = bin_edges[index]
                tmp = sf[sf[flux_freq] > flux1]
                tmp = tmp[tmp[flux_freq] <= flux2]
                if len(tmp) == 0:
                    continue
                index = np.random.choice(tmp.index)
                ggg = tmp.loc[index, "galaxy"]
                if ggg in list(galaxies["galaxy"]):
                    continue
                galaxies = galaxies.append({"ra": tmp.loc[index, "ra"],
                                        "dec": tmp.loc[index, "dec"],
                                        "i_151": tmp.loc[index, "i_151"],
                                        flux_freq: tmp.loc[index, flux_freq],
                                        "type": tmp.loc[index, "type"],
                                        "galaxy": tmp.loc[index, "galaxy"],
                
                }, ignore_index=True) 


            else:
                agntype = "SB"
                flux1 = bin_edges[index-1]
                flux2 = bin_edges[index]
                tmp = sb[sb[flux_freq] > flux1]
                tmp = tmp[tmp[flux_freq] <= flux2]
                if len(tmp) == 0:
                    continue
                index = np.random.choice(tmp.index)
                ggg = tmp.loc[index, "galaxy"]
                if ggg in list(galaxies["galaxy"]):
                    continue
                galaxies = galaxies.append({"ra": tmp.loc[index, "ra"],
                                        "dec": tmp.loc[index, "dec"],
                                        "i_151": tmp.loc[index, "i_151"],
                                        flux_freq: tmp.loc[index, flux_freq],
                                        "type": tmp.loc[index, "type"],
                                        "galaxy": tmp.loc[index, "galaxy"],
                
                }, ignore_index=True)
            if len(galaxies) / 100 == 0:
                print(len(galaxies))
        return galaxies
    
    def sampling2(self, number_of_galaxies, gtype, freq, flux_min, flux_max, fov):
        flux_freq = "i_" + str(freq)
        fovl, fovu = fov
        if gtype == 'RQ':
            if flux_freq not in self.rqdb.keys():
                self.rqdb[flux_freq] = rqq_spec(self.rqdb["i_151"], freq)
            rq = self.rqdb[(self.rqdb["ra"]>fovl) & (self.rqdb["ra"]<fovu) \
                     & (self.rqdb["dec"]>fovl) & (self.rqdb["dec"]<fovu)]
            flux_min = np.max([np.min(rq[flux_freq]), flux_min])
            flux_max = np.min([flux_max, np.max(rq[flux_freq])])
            print("Minimum flux: %.2e; Maximum flux: %.2e." %(flux_min, flux_max))
            rq = rq[(rq[flux_freq] >= flux_min) & (rq[flux_freq] <= flux_max)]
            galaxies = rq
            if number_of_galaxies == -1:
                return galaxies
            
        if gtype == 'FR1': 
            if flux_freq not in self.fr1.keys():
                self.fr1db[flux_freq] = fr1_core_spec( self.fr1db["core_i_151"], freq ) \
                    + fr1_lobe_spec( self.fr1db["lobe1_i_151"], freq) \
                    + fr1_lobe_spec( self.fr1db["lobe2_i_151"], freq)
            fr1 = self.fr1db[(self.fr1db["ra"]>fovl) & (self.fr1db["ra"]<fovu) \
                       & (self.fr1db["dec"]>fovl) & (self.fr1db["dec"]<fovu)]
            flux_min = np.max([np.min(fr1[flux_freq]), flux_min])
            flux_max = np.min([flux_max, np.max(fr1[flux_freq])])
            print("Minimum flux: %.2e; Maximum flux: %.2e." %(flux_min, flux_max))
            fr1 = fr1[(fr1[flux_freq] >= flux_min) & (fr1[flux_freq] <= flux_max)]
            galaxies = fr1
            if number_of_galaxies == -1:
                return galaxies
            
        if gtype == 'FR2': 
            if flux_freq not in self.fr2.keys():
                self.fr2db[flux_freq] = fr2_core_spec( self.fr2db["core_i_151"], freq ) \
                    + fr2_lobe_spec( self.fr2db["lobe1_i_151"], freq) \
                    + fr2_lobe_spec( self.fr2db["lobe2_i_151"], freq)\
                    + fr2_hotspot_spec( self.fr2db["hotspot1_i_151"], freq) \
                    + fr2_hotspot_spec( self.fr2db["hotspot2_i_151"], freq)
            fr2 = self.fr2db[(self.fr2db["ra"]>fovl) & (self.fr2db["ra"]<fovu) \
                       & (self.fr2db["dec"]>fovl) & (self.fr2db["dec"]<fovu)]
            flux_min = np.max([np.min(fr2[flux_freq]), flux_min])
            flux_max = np.min([flux_max, np.max(fr2[flux_freq])])
            print("Minimum flux: %.2e; Maximum flux: %.2e." %(flux_min, flux_max))
            fr2 = fr2[(fr2[flux_freq] >= flux_min) & (fr2[flux_freq] <= flux_max)]
            galaxies = fr2
            if number_of_galaxies == -1:
                return galaxies
            
        if gtype == 'SF':
            if flux_freq not in self.sf.keys():
                self.sfdb[flux_freq] = sf_spec(self.sfdb["i_151"], freq)
            sf = self.sfdb[(self.sfdb["ra"]>fovl) & (self.sfdb["ra"]<fovu) \
                     & (self.sfdb["dec"]>fovl) & (self.sfdb["dec"]<fovu)]
            flux_min = np.max([np.min(sf[flux_freq]), flux_min])
            flux_max = np.min([flux_max, np.max(sf[flux_freq])])
            print("Minimum flux: %.2e; Maximum flux: %.2e." %(flux_min, flux_max))
            sf = sf[(sf[flux_freq] >= flux_min) & (sf[flux_freq] <= flux_max)]
            galaxies = sf
            if number_of_galaxies == -1:
                return galaxies
            
        if gtype == 'SB': 
            if flux_freq not in self.sb.keys():
                self.sbdb[flux_freq] = sb_spec(self.sbdb["i_151"], freq)
            sb = self.sbdb[(self.sbdb["ra"]>fovl) & (self.sbdb["ra"]<fovu) \
                     & (self.sbdb["dec"]>fovl) & (self.sbdb["dec"]<fovu)]
            flux_min = np.max([np.min(sb[flux_freq]), flux_min])
            flux_max = np.min([flux_max, np.max(sb[flux_freq])])
            print("Minimum flux: %.2e; Maximum flux: %.2e." %(flux_min, flux_max))
            sb = sb[(sb[flux_freq] >= flux_min) & (sb[flux_freq] <= flux_max)]
            galaxies = sb
            if number_of_galaxies == -1:
                return galaxies
        
        if number_of_galaxies > len(galaxies):
            raise ValueError("The number of %s galaxies can not be greater than %d." %(gtype, len(galaxies)))
        index_selected = random.sample(galaxies.index.tolist(), number_of_galaxies)
        return galaxies.loc[index_selected]