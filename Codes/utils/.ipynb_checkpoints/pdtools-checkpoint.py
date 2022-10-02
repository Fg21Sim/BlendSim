# Copyright (c) 2021-2022 Chenxi Shan <cxshan@hey.com>

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def common( series ):
    """
    common display the most frequent items from the series
    :Params series: input series;
    :Output freq: output freq series;
    """
    freq = series.value_counts()
    print("Printing the frequency")
    display(freq)
    
    return freq


def get_unique( series, n, direct='Nope' ):
    """
    Get the unique total_flux from the WilmanDB for sources with multiple comps
    
    :Params series: the pd series object of the total_flux 
    :Params n: number of params
    :Params direct: Direct flag, if direct != 'Nope', use unique()
    :Output uni_series: the output unique series
    """
    if direct == 'Nope':
        if n == 1:
            uni_series = series
        else:
            uni = []
            for i in range(0,len(series),n):
                a = series[i]
                uni.append(a)
            uni_series = pd.Series(uni)
    else:
        uni_series = series.unique()
        
    return uni_series


def gen_bins(b, nbins):
    """
    gen flux bins in log space;
    :Params b: flux bins limits;
    :Params nbins: # of bins;
    :Output bins: the output bin_edges;
    """
    if isinstance(b, list) and len(b) == 2:
        bmin = b[0]
        bmax = b[1]
        print(bmin, bmax)
    else:
        print('The bin lim is not set properly! Need to be len(list)=2.')
    bins = np.logspace(np.log10(bmin), np.log10(bmax), nbins)
    print(bins)
    
    return bins


def stat( db, flux_bins, fov='Nope' ):
    if fov != 'Nope':
        db_fov = db[(db['ra']>=fov[0]) & (db['ra']<=fov[1]) 
                   & (db['dec']>=fov[0]) & (db['dec']<=fov[1])]
    else:
        db_fov = db
    hist, bin_edges = np.histogram(db_fov, flux_bins)
    
    return [hist, bin_edges]


def statuni( db, flux_bins, n , fov='Nope' ):
    if fov != 'Nope':
        db_fov = db[(db['ra']>=fov[0]) & (db['ra']<=fov[1]) 
                   & (db['dec']>=fov[0]) & (db['dec']<=fov[1])]
    else:
        db_fov = db
    print('======= Unique Starts =======')
    data = get_unique( db_fov['i_151_tot'], n )
    print('======= Unique Ends =========')    
    hist, bin_edges = np.histogram(data, flux_bins)
    
    return [hist, bin_edges]


def statuni_ploth( db, b, nbins, n, title, fov='Nope' ):
    if fov != 'Nope':
        db_fov = db[(db['ra']>=fov[0]) & (db['ra']<=fov[1]) 
                   & (db['dec']>=fov[0]) & (db['dec']<=fov[1])]
    else:
        db_fov = db
    print('======= Unique Starts =======')
    data = get_unique( db_fov['i_151_tot'], n )
    print('======= Unique Ends =========')
    # Gen bins
    flux_bins = gen_bins( b, nbins )
    hist, bin_edges = np.histogram( data, flux_bins )
    # Plotting
    dplothlog( data, title, flux_bins, w=10, h=10 )
    
    return [hist, bin_edges]


def gen_DBstat( DB, flux_bins, types=['rq'], fov='Nope' ):
    """
    *** gen_DBstat() calculates the per flux bin source numbers given DB object ***
    *** Histogram per type is displayed ***
    
    :Params DB: Database object from WilmanDB();
    :Params flux_bins: bins from gen_bins();
    :Params types: the types are calculated;
    :Params fov: fov filter to pass to stat();
    ::::::::::::
    :Output DBstat: DBstat dictionary with flux_bins, types, & hists;
    """
    DBstat = {}
    DBstat['types'] = types
    types = types
    flux_bins = flux_bins
    DBstat['bins'] = flux_bins
    
    for i in types:
        db = getattr(DB, i)
        n = db['components'][0]
        flux = db['i_151_tot']
        print('=======', i, 'Unique Starts =======')
        flux_uni = get_unique(db['i_151_tot'], n)
        print('=======', i, 'Unique Ends =========')
        st = stat(flux_uni, flux_bins, fov)
        dplothlog(flux_uni, i + ' Flux Hist', flux_bins)
        
        keys = [ i+'_uni', i+'_hist', i+'_bins' ]
        
        DBstat[i] = {}
        DBstat[i]['db'] = db
        DBstat[i][keys[0]] = flux_uni
        DBstat[i][keys[1]] = st[0]
        DBstat[i][keys[2]] = st[1]
    return DBstat


def flux_cata( DBstat, verbose=False ):
    """
    Convert the DBstat object from gen_DBstat() to a flux_cata object
    :Params DBstat: input object;
    :Output flux_cata: flux_cata object to supply the survey
    """
    flux_cata = {}
    flux_cata['bins'] = DBstat['bins']
    category_names = DBstat['types']
    flux_cata['category_names'] = category_names
    bins = DBstat['bins']
    results = {}
    for i in range(1,len(bins)):
        if verbose == True:
            print(i, 'bin edge:', bins[i])
        results[str(bins[i])] = []
        for j in category_names:
            a = DBstat[j][j+'_hist'][i-1]
            if verbose == True:
                print('# of', j, a)
            results[str(bins[i])].append(a)
    flux_cata['results'] = results
    return flux_cata


def gen_ratios( flux_cata ):
    """
    *** Output ratios ***
    
    :Params flux_cata: flux_cata dict from flux_cata();
    :Output flux_cata: add flux_cata with ratios;
    """
    results = flux_cata['results']
    ratio = {}
    
    keys = list(results.keys())
    for i in range(len(results)):
        list_ratio = []
        for j in results[keys[i]]:
            r = j / sum(results[keys[i]])
            list_ratio.append(r)
        ratio['bin'+str(i)] = list_ratio
        
    flux_cata['ratio'] = ratio
    
    return flux_cata

#=================== Plotting ======================

def dploth( data, title, bins, w=10, h=10 ):
    plt.figure(figsize=(w,h))
    plt.title(title)
    plt.hist(data, bins=bins)
    plt.show()
    
    
def dplothlog( data, title, bins, w=10, h=10 ):
    plt.figure(figsize=(w,h))
    plt.title(title)
    plt.hist(data, bins=bins)
    plt.xscale("log")
    plt.show()
    
    
def dplothlogbin( data, title, b, nbins, w=10, h=10 ):
    bins = gen_bins( b, nbins )
    plt.figure(figsize=(w,h))
    plt.title(title)
    plt.hist(data, bins=bins)
    plt.xscale("log")
    plt.show()
    
    
def dploth_multi( DBstat, log=True ):
    """
    *** Plot the multi-histogram of the DBstat object from gen_DBstat() ***
    :Params DBstat: input object;
    !!! Add color in the future !!!
    """
    bins = DBstat['bins']
    types = DBstat['types']
    
    # Gen a list to contain all the data sets
    multi = []
    for i in types:
        a = DBstat[i][i+'_uni']
        #print(type(a))
        a = a.to_numpy()
        #print(type(a))
        multi.append(a)
    
    plt.figure(figsize=(10,10))
    plt.hist(multi, bins, histtype='bar', stacked=True)
    plt.legend(types)
    if log == True:
        plt.xscale("log")
        plt.yscale("log")
    plt.show()
    
    
def dplotbarh(results, category_names, log=True, demo=False, text=False):
    """
    *** Plot the horizontal bar chart of a flux_cata object {'results':results, 'category_names':category_names} ***
    *** Ref: https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html ***
    
    ----------
    Params: results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    Params: category_names : list of str
        The category labels.
    Output: fig, ax
    """
    
    if demo == True:
        category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
        results = {
            'Question 1': [10, 15, 17, 32, 26],
            'Question 2': [26, 22, 29, 10, 13],
            'Question 3': [35, 37, 7, 2, 19],
            'Question 4': [32, 11, 9, 15, 33],
            'Question 5': [21, 29, 5, 5, 40],
            'Question 6': [8, 19, 5, 30, 38]
        }
    else:
        results = results
        category_names = category_names
    
    labels = list(results.keys())
    
    data = np.array(list(results.values()))
    
    #print(data, type(data), data.shape)
    data_cum = data.cumsum(axis=1)
    #print(data_cum, type(data_cum))
    
    # Gen discrete colors
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))
    #print(category_colors)
    
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False) # Hide the xaxis ticks & labels if open 
    if log == True:
        plt.xscale("log")
        
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        
        if text == True:
            # Add text label to the bar!
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            text_x = starts + widths / 2
            text_y = ax.get_yticks()
            for text_i, text_yi, width in zip(text_x, text_y, widths):
                ax.text(text_i, text_yi, width, color=text_color, ha='center', va="center")
        
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    plt.show()
    return fig, ax