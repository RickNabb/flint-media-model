import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
import pandas as pd
import re
import array as arr
from scipy.stats import pearsonr


#To run this- run it in flint-media-model\data\try2\with-media
    #this has the google files and other files in it
    #add in the csv that has to be processed

#Load in google csv- only used for figures (won't be called in the main function)
def load_google_data(google_csv):
    google_trends = pd.read_csv(google_csv)
    google_trends.columns = ['week', 'data']
    google_data = google_trends['data'].to_list()
    return google_data

#load in influence model csv- make sure column names match
def load_infl_model(csv):
    infl_mod = pd.read_csv(csv)
    infl_mod.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-media-influence',
                  'citizen-citizen-influence','repetition', 'data', 'num_media']
    return infl_mod

#load in org model csv- make sure column names match
def load_org_mod(csv):
    org_mod = pd.read_csv(csv)
    org_mod.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m',
                       'citizen-media-influence',
                       'citizen-citizen-influence', 'organizing-capacity', 'flint-organizing-strategy', 'repetition',
                       'data', 'num_media']
    return org_mod

#some of the processing to properly interpret the data column
def convertdata(data):
    sdata = data.strip(' ')
    ndata = nlogo_parse_chunk(sdata)
    n2data = [elem.replace('.', '') for elem in ndata]
    list=[]
    for x in n2data:
        list.append(x.replace("\r\n", ""))
    list = [i for i in list if i]
    return list

#netlogo processing from how they output
def nlogo_parse_chunk(chunk):
    chunk = chunk.strip().replace('"','')
    if chunk.count('[') > 1 and chunk[0] == '[':
        return nlogo_mixed_list_to_dict_rec(chunk[chunk.index('['):].strip())
    elif chunk.count('[') > 1 or chunk[0] != '[':
        return { chunk[0:chunk.index('[')].strip(): nlogo_mixed_list_to_dict_rec(chunk[chunk.index('['):].strip()) }

    pieces = chunk.strip().replace('[','').replace(']','').split(' ')
    if len(pieces) == 2:
        return { pieces[0]: pieces[1] }
    else:
        return pieces


def convert_to_int(data):
    for i in range(0, len(data)):
        data[i] = int(data[i])
    return data

#use this function to make the new table with metrics
def make_df_with_metrics(csv):

    #change line below depending if it is infl or org version
    #for influence:
    infl_mod = load_infl_model(csv)

    #for org
    #infl_mod = load_org_mod(csv)


    infl_mod["short list"] = None
    infl_mod['unscaled-list'] = None
    # threshold is a binary value representing whether or not the critical mass of the population has been reached (represented by
    # 70%)
    infl_mod['threshold'] = None
    # peak-time is the time at which the peak will occur
    infl_mod['peak-time'] = None
    infl_mod['1-week-spread']=None
    # total spread is how many agents were reached in sum
    infl_mod['total-spread'] = None
    for i in range(0, infl_mod.shape[0]):
        thresh = 0
        error = 0
        delta_max_1 = 0
        cutoff = (300 + int(infl_mod['num_media'].iloc[i])) * 0.7

        run1 = infl_mod["data"].iloc[i]
        finallist = convertdata(run1)
        int_data = convert_to_int(finallist)
        short_list = int_data[: 114]
        infl_mod['unscaled-list'].iloc[i] = short_list
        scaled_short_list = []
        # code below scales data from 0-100 to match google trends- this is needed to calculate total error, pearson's r, etc
        if max(short_list) >= 1:
            for k in range(0, len(short_list)):
                val = (short_list[k] / max(short_list)) * 100
                scaled_short_list.append(val)
            time_of_peak = short_list.index(max(short_list))
        else:
            for k in range(0, len(short_list)):
                val = (short_list[k] / 1) * 100
                scaled_short_list.append(val)
        infl_mod['short list'].iloc[i] = scaled_short_list
        for h in range(0, len(scaled_short_list)):
            thresh = thresh + int_data[h]
        # since thresh is untransformed, we tabulate off of raw data cut to 114 weeks, not the scaled 0-100 data
        infl_mod['total-spread'].iloc[i] = thresh
        if thresh >= cutoff:
            infl_mod['threshold'].iloc[i] = 1
        else:
            infl_mod['threshold'].iloc[i] = 0
        for a in range(1, len(scaled_short_list)):
            delta_1 = scaled_short_list[a] - scaled_short_list[a - 1]
            if delta_1 > delta_max_1:
                delta_max_1 = delta_1
            else:
                pass

        infl_mod['peak-time'].iloc[i] = time_of_peak
        infl_mod['1-week-spread'].iloc[i] = delta_max_1
        # 2-week spread

        if i % 5000 == 0:
            print(i)
#adjust this to the name of the csv you want
    infl_mod.to_csv('influence_mod_with_metrics_test.csv')
    return infl_mod

make_df_with_metrics('static-infl-sweep-test-for-nick.csv')


