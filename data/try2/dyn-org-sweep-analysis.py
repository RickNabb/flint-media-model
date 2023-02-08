import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import sklearn
import os
from sklearn.decomposition import PCA
os.environ["PATH"] += os.pathsep + 'C:/Users/cknox/anaconda3/envs/flintdata2/Library/bin/graphviz'
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import array as arr
from scipy.stats import pearsonr

'''
NETLOGO PARSING
'''

def nlogo_list_to_arr(list_str):
    return [ el.replace('[', '').strip().split(' ') for el in list_str[1:len(list_str)-1].split(']') ]

#def nlogo_replace_agents(string, types):
#    for type in types:
#        string = string.replace(f'({type} ', f'{type}_')
#    return string.replace(')','')
'''
Parse a NetLogo mixed dictionary into a Python dictionary. This is a nightmare.
But it works.

:param list_str: The NetLogo dictionary as a string.
'''
def nlogo_mixed_list_to_dict(list_str):
  return nlogo_parse_chunk(list_str)

def nlogo_mixed_list_to_dict_rec(list_str):
  # print(f'processing {list_str}')
  if list_str[0] == '[' and list_str[len(list_str)-1] == ']' and list_str.count('[') == 1:
    return nlogo_parse_chunk(list_str)

  d = {}
  chunk = ''
  stack_count = 0
  for i in range(1, len(list_str)-1):
    chunks = []
    char = list_str[i]
    chunk += char
    if char == '[':
      stack_count += 1
    elif char == ']':
      stack_count -= 1

      if stack_count == 0:
        # print(f'parsing chunk: {chunk}')
        parsed = nlogo_parse_chunk(chunk)
        # print(f'parsed: {parsed}')
        d[list(parsed.keys())[0]] = list(parsed.values())[0]
        chunk = ''
      # chunks[stack_count] += char
  #print(d)
  return d

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


def createdataframe(dataset):
    df = pd.read_csv(dataset)
    df.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'organizing-capacity',
                  'organizing-strategy','repetition', 'data']
    print(df)
    return df

def convertdata(data):
    sdata = data.strip(' ')
    ndata = nlogo_parse_chunk(sdata)
    n2data = [elem.replace('.', '') for elem in ndata]
    list=[]
    for x in n2data:
        list.append(x.replace("\r\n", ""))
    list = [i for i in list if i]
    return list

def convert_to_int(data):
    for i in range(0, len(data)):
        data[i] = int(data[i])
    return data

def convert_to_float(data):
    for i in range(0, len(data)):
        data[i] = float(data[i])
    return data

def loop_per_row(df_adj, google_trends_data):
    google_trends_data=google_trends_data
    google_trends_data=google_trends_data[: 100]
    max_val = 1
    df_new=df_adj.copy()
    df_new['threshold'] = None
    df_new['rmse']=None
    df_new['total_error']=None
    df_new['corr_coef']=None
    for i in range(0,df_new.shape[0]):
        thresh=0
        error=0
        cutoff=150
        run1 = df_new["data"].iloc[i]
        #data = run1['data']
        finallist = convertdata(run1)
        int_data = convert_to_int(finallist)
        short_list=int_data[: 101]
        for h in range(0, len(short_list)):
            thresh=thresh+int_data[h]
            error_addition=abs(short_list[h]-google_trends_data[h])
            error=error+error_addition
        if thresh >= cutoff:
            df_new['threshold'].iloc[i] = 1
            print("thresh > cutoff", i)
        else:
            df_new.iat[i, 11] = 0
            print('nope',i)
        corr_coef=pearsonr(google_trends_data,short_list)
        corr_val=corr_coef[0]
        df_new['corr_coef'].iloc[i] = corr_val
        RMSE = (mean_squared_error(google_trends_data, short_list, squared=True))
        df_new['rmse'].iloc[i]=RMSE
        df_new["total_error"].iloc[i]=error
        print('error', error)
    df_new.to_csv("dyn-org-sweep-eval.csv")


    print(df_new)







        #old stuff is below
 #       for i in range(0, len(int_data)):
 #           max_val_this_run = max(int_data)
 #           thresh=thresh+int_data[i]
 #           if max_val_this_run >= max_val:
 #               max_val=max_val_this_run
 #           else:
 #               pass
 #           print('max_val', max_val)
 #       df.iat[i,1]=thresh
 #   for i in range(0, df.shape[0]):
 #       adj_data = []
 #       run = df_with_dtw.iloc[i]
 #       data = run['data']
 #       finallist = convertdata(data)
 #       int_data = convert_to_int(finallist)
 #       short_list = int_data[:114]
 #       for i in range(0, len(int_data)):
 #           adj_val = (int_data[i] / (max_val)) * 100
 #           adj_data.append(adj_val)
 #       tuple_list=[]
 #       for i in range(0, len(adj_data)):
 #           element=[]
 #           element.append(adj_data[i])
 #           element.append(i)
 #           tuple_list.append(element)
 #       RMSE = (mean_squared_error(google_trends_data, short_list, squared=True))
 #       corr_coef = pearsonr(google_trends_data,short_list)
 #       corr_coef=corr_coef[0]
 #       print('cc', corr_coef )
 #       #print(RMSE)
 #       rsme_list.append(RMSE)
 #       corr_coef_list.append(corr_coef)
 #       #print(tuple_list)
 #   for i in range(0, len(dtw_list)):
 #       dtw_val=dtw_list[i]
 #       #note as the original dataframe shape changes, the value below can be 9,10,11, etc
 #       df_with_dtw.iat[i,10]=dtw_val
 #       rsme_val=rsme_list[i]
 #       df_with_dtw.iat[i,11]=rsme_val
 #       corr_val=corr_coef_list[i]
 #       df_with_dtw.iat[i, 12] = corr_val
 #       #print(max(short_list))
 #       #now we need to add columns for RMSE, MSE, BIAS, VARIANCE AND THEN RETURN THE DATAFRAME
 #       #example of how to add column below
 #      #df.at[i,'class-time'] = class_of_peak_time
 #   print(df_with_dtw)
 #   return df_with_dtw
 #       #here we need to start getting metrics

def load_google(file):
    df = pd.read_csv(file)
    df.columns = ['week', 'data']
    data = df['data'].to_list()
    print(data)
    return data


#this is the code for csv 'media-connections-dynamic-organizing-exp-results
def analyze(csv_file):
    # load data
    google_trends_data = load_google("google-trends.csv")
    print(google_trends_data)
    df_adj=createdataframe(csv_file)
    dataframe_with_metrics = loop_per_row(df_adj, google_trends_data)
    print(dataframe_with_metrics)
    #cut data- this is a big assumption- would have to do a different analysis to make this flexible
    return dataframe_with_metrics
    #take metrics (mse, rmse, bias, variance)

def best_graph(results):
    df = pd.read_csv("google-trends.csv")
    df.columns = ['week', 'data']
    google_data = df['data'].to_list()
    run1 = results.iloc[569]
    data = run1['data']
    finallist = convertdata(data)
    int_data = convert_to_int(finallist)
    max_val = 0
    for i in range(0, len(int_data)):
        max_val_this_run = max(int_data)
        if max_val_this_run >= max_val:
            max_val = max_val_this_run
        else:
            pass
    adj_data = []
    for i in range(0, len(int_data)):
        adj_val = (int_data[i] / (max_val)) * 100
        adj_data.append(adj_val)
    #NEED TO PROCESS THIS DATA
    plt.plot(google_data, label="google trends")
    plt.plot(adj_data, label = "best simulation")
    plt.legend()
    plt.title("Closest Model Based on DTW influence model sweep.")
    plt.show()


    pass
    #here make best results off of whatever row specified......

def df_figures(results):
    sns.boxplot(data=results, x='citizen-media-influence', y='dtw')
    # sns.scatterplot(data=df_with_class, x='class-height', y='class-time')
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.xlabel('Citizen-Media-Influence')
    plt.ylabel('DTW Distance')
    plt.title('media-con-dyn-org')
    plt.legend(loc='upper right')
    plt.show()


def main(csv_file):
    results = analyze(csv_file)
    #print("THIS ROW dtw",results[results.dtw == results.dtw.min()])
    #print("THIS ROW RSME", results[results.dtw == results.dtw.min()])
    #results.to_csv('media-connections-dynamic-organizing-exp-results_dtw.csv')
    #best_graph(results)
    #df_figures(results)
    #data_med_con_dyn_org = results['dtw'].to_list()
    #plt.hist(data_med_con_dyn_org, bins=10)
    #plt.title("dtw Data_med_con_dyn_org")
    #plt.show()

main('dynamic-organizing-sweep-exp-results.csv')
