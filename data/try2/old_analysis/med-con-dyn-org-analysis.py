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
                  'organizing-strategy', 'repetition', 'data']
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
    max_val = 1
    df_new=df_adj.copy()
    df_new["short list"]=None
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
        short_list=int_data[: 114]
        scaled_short_list=[]
        #code below adds in scaled impact
        if max(short_list)>= 1:
            for k in range(0, len(short_list)):
                val=(short_list[k]/max(short_list))*100
                scaled_short_list.append(val)
        else:
            for k in range(0, len(short_list)):
                val = (short_list[k] / 1) * 100
                scaled_short_list.append(val)
        #df_new['short list'].iloc[i]=short_list
        df_new['short list'].iloc[i] = scaled_short_list
        for h in range(0, len(scaled_short_list)):
            thresh=thresh+int_data[h]
            error_addition=abs(scaled_short_list[h]-google_trends_data[h])
            error=error+error_addition
        if thresh >= cutoff:
            df_new['threshold'].iloc[i] = 1
            print("thresh > cutoff", i)
        else:
            df_new['threshold'].iloc[i] = 0
            print('nope',i)
        corr_coef=pearsonr(google_trends_data,scaled_short_list)
        corr_val=corr_coef[0]
        df_new['corr_coef'].iloc[i] = corr_val
        RMSE = (mean_squared_error(google_trends_data, scaled_short_list, squared=True))
        df_new['rmse'].iloc[i]=RMSE
        df_new["total_error"].iloc[i]=error
        print('error', error)

        #create pandas dataframe for each over threshold

    #need to sort these by threshold- also no transformation is happening here
    #test a few?
    df_new = df_new[df_new["threshold"] == 1]
    df_new.to_csv("med-con-dyn-org-sweep-analysis.csv")
    sns.scatterplot(data=df_new, x='rmse', y='corr_coef')
    plt.title("Dynamic Organizing Sweep")
    plt.xlabel("RMSE")
    plt.ylabel("Corr_coef")
    plt.show()

    min_RMSE=df_new[df_new.rmse == df_new.rmse.min()]
    print('min rmse:',min_RMSE)
    run = min_RMSE["short list"].tolist()
    run=run[0]
    time_list=[]
    for i in range(0, 114):
        time_list.append(i)
    print(time_list)
    print(run)
    print(type(run))
    plt.plot(time_list, run, label='simulation')
    plt.plot(time_list, google_trends_data, label= 'google trends')
    plt.legend()
    plt.title("Min RMSE Media & Dynamic Organizing Sweep")
    plt.show()

    max_cc = df_new[df_new.corr_coef == df_new.corr_coef.max()]
    print('max cc:', max_cc)
    run = max_cc["short list"].tolist()
    run = run[0]
    time_list = []
    for i in range(0, 114):
        time_list.append(i)
    print(time_list)
    print(run)
    print(type(run))
    plt.plot(time_list, run, label='simulation')
    plt.plot(time_list, google_trends_data, label='google trends')
    plt.legend()
    plt.title("Max CC Media & Dynamic Organizing Sweep")
    plt.show()

    max_RMSE=df_new[df_new.rmse == df_new.rmse.max()]
    print('min rmse:',max_RMSE)
    run = max_RMSE["short list"].tolist()
    run=run[0]
    time_list=[]
    for i in range(0, 114):
        time_list.append(i)
    print(time_list)
    print(run)
    print(type(run))
    plt.plot(time_list, run, label='simulation')
    plt.plot(time_list, google_trends_data, label= 'google trends')
    plt.legend()
    plt.title("Max RMSE Media & Dynamic Organizing Sweep")
    plt.show()

    print("average RMSE:", df_new.rmse.mean())
    print("average corr_coef:", df_new.corr_coef.mean())
    print("average total_error:", df_new.total_error.mean())


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

main('media-connections-dynamic-organizing-exp-results.csv')
