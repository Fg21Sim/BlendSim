class Simulate:
    def __init__(self, Params):
        self.Params = Params
        self.skymodel = []
        self.galaxies_simulated = []
        self.total_number_of_simulated_galaxies = 0
        self.n_rq = 0
        self.n_fr1 = 0
        self.n_fr2 = 0
        self.n_sf = 0
        self.n_sb = 0
        self.init_switch()
        self.rsg = RSgen(self.Params)
    
    def init_switch(self):
        """
        Initialize simulation switches.
            0/False: No
            1/Ture: Yes
        """
        self.rq_switch = 1 
        self.fr1_switch = 1  
        self.fr2_switch = 1  
        self.sf_switch = 1 
        self.sb_switch = 1
        if self.Params["number_of_rq"] == 0:
            self.rq_switch = False
        if self.Params["number_of_fr1"] == 0:
            self.fr1_switch = False
        if self.Params["number_of_fr2"] == 0:
            self.fr2_switch = False
        if self.Params["number_of_sf"] == 0:
            self.sf_switch = False
        if self.Params["number_of_sb"] == 0:
            self.sb_switch = False
        return 0
    
    def check_distance(self, x, y):
        """
        Check whether the distance between the new source and \
        other simulated sources is greater than dmin.
        Return: 
            True if the distance is greater than dmin, otherwise return False.
        """
        for galaxy in self.skymodel:
            distance = np.sqrt((galaxy[:, 0] - x) ** 2 + (galaxy[:, 1] - y) ** 2)
            if np.sum(distance < self.Params['dmin']) > 1:
                return False
            else:
                return True
        
    def make_skymodel(self, galaxies):
        for i in galaxies.index:
            if galaxies.loc[i]['type'] == 'RQQ':
                if self.rq_switch:
                    g = DB.rq.loc[DB.rq['galaxy'] == galaxies.loc[i]['galaxy']]
                    pixels = self.rsg.rq(g)
                else:
                    continue
            if galaxies.loc[i]['type'] == 'FR1':
                if self.fr1_switch:
                    g = DB.fr1.loc[DB.fr1['galaxy'] == galaxies.loc[i]['galaxy']]
                    pixels = self.rsg.fr1Wilman(g)
                else:
                    continue
            if galaxies.loc[i]['type'] == 'FR2':
                if self.fr2_switch:
                    g = DB.fr2.loc[DB.fr2['galaxy'] == galaxies.loc[i]['galaxy']]
                    pixels = self.rsg.fr2Wilman(g)
                else:
                    continue
            if galaxies.loc[i]['type'] == 'SF':
                if self.sf_switch:
                    g = DB.sf.loc[DB.sf['galaxy'] == galaxies.loc[i]['galaxy']]
                    pixels = self.rsg.sfsb(g)
                else:
                    continue
            if galaxies.loc[i]['type'] == 'SB':
                if self.sb_switch:
                    g = DB.sb.loc[DB.sb['galaxy'] == galaxies.loc[i]['galaxy']]
                    pixels = self.rsg.sfsb(g)
                else:
                    continue

            if galaxies.loc[i]['galaxy'] in set(self.galaxies_simulated):
                continue
            is_greater_than_dmin = True
            if len(self.galaxies_simulated) > 1 and self.Params['dmin'] > 0:
                for pix in pixels:
                    x = pix[0]
                    y = pix[1]
                    if not self.check_distance(x, y):
                        is_greater_than_dmin = False
                        continue           
                if is_greater_than_dmin:
                    self.skymodel.append(np.array(pixels))
                    self.galaxies_simulated.append(galaxies.loc[i]['galaxy'])
                    if galaxies.loc[i]['type'] == 'RQQ':
                        self.n_rq += 1
                    if galaxies.loc[i]['type'] == 'FR1':
                        self.n_fr1 += 1              
                    if galaxies.loc[i]['type'] == 'FR2':
                        self.n_fr2 += 1 
                    if galaxies.loc[i]['type'] == 'SF':
                        self.n_sf += 1
                    if galaxies.loc[i]['type'] == 'SB':
                        self.n_sb += 1
            else:
                self.skymodel.append(np.array(pixels))
                self.galaxies_simulated.append(galaxies.loc[i]['galaxy'])
                if galaxies.loc[i]['type'] == 'RQQ':
                    self.n_rq += 1
                if galaxies.loc[i]['type'] == 'FR1':
                    self.n_fr1 += 1              
                if galaxies.loc[i]['type'] == 'FR2':
                    self.n_fr2 += 1 
                if galaxies.loc[i]['type'] == 'SF':
                    self.n_sf += 1
                if galaxies.loc[i]['type'] == 'SB':
                    self.n_sb += 1
            self.total_number_of_simulated_galaxies = self.n_rq + self.n_fr1 + self.n_fr2 + self.n_sb + self.n_sf
            
            if self.n_rq >= self.Params["number_of_rq"]:
                self.rq_switch = False
            if self.n_fr1 >= self.Params["number_of_fr1"]:
                self.fr1_switch = False
            if self.n_fr2 >= self.Params["number_of_fr2"]:
                self.fr2_switch = False
            if self.n_sf >= self.Params["number_of_sf"]:
                self.sf_switch = False
            if self.n_sb >= self.Params["number_of_sb"]:
                self.sb_switch = False
            
            print("Total number of simulated galaxies: %d" %(self.total_number_of_simulated_galaxies))
            print("RQ: %d; FR1: %d; FR2: %d; SB: %d; SF: %d." %(self.n_rq, self.n_fr1, self.n_fr2, self.n_sb, self.n_sf))
        return self.skymodel