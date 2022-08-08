import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.signal import savgol_filter
import numpy as np
import tslearn.clustering
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import graphviz
import os
from sklearn.decomposition import PCA
os.environ["PATH"] += os.pathsep + 'C:/Users/cknox/anaconda3/envs/flintdata2/Library/bin/graphviz'

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



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
    df =  pd.read_csv(dataset, sep='|' , engine='python')
    #print(df)
    df.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence', 'flint-community-size', 'data']
    df.set_index('run', inplace=True)
    return df

def convertdata(data):
    print(data)
    sdata = data.strip(' ')
    #print('stripped data', sdata)
    ndata = nlogo_parse_chunk(sdata)
    #print(ndata)
    n2data = [elem.replace('.', '') for elem in ndata]
    #print(len(n2data))
    list=[]
    for x in n2data:
        list.append(x.replace("\r\n", ""))
    #print(list)
    list = [i for i in list if i]
    #print(list)
    #print(len(list))
    return list

def convert_to_int(data):
    for i in range(0, len(data)):
        data[i] = int(data[i])
    return data

def apply_filter(int_data):
    #newdata = savgol_filter(int_data, 15, 2)
    newdata = savgol_filter(int_data, 15, 2)
    return newdata
    #could define spike as x% of population in x time steps or use filter and check peak

def loop_per_row(df):
    df.insert(0, "class", " ")
    for i in range(0,df.shape[0]):
        run1 = df.iloc[i]
        data = run1['data']
        finallist = convertdata(data)
        int_data = convert_to_int(finallist)
        #print(int_data)
        '''for full dataframe'''
        class_of_peak = evaluate_peak(int_data)
        df.at[i,'class'] = class_of_peak
    return(df)

        #ylength = len(int_data)
        #USE TIME TO GRAPH
        #time = []
        #for i in range(1, ylength + 1):
        #    timestep = i
        #    time.append(timestep)
    '''Code below is for filtered data'''
    #filtered_int = apply_filter(int_data)
    #return(filtered_int, time)
    #evaluate_peak(filtered_int, time)
    '''For non-filtered data, use below:'''
        #class_of_peak = evaluate_peak(int_data)
    # return(int_data, time)
    '''To only return class for one trial'''
        #return(class_of_peak)




def evaluate_peak(data):
    deltas=[]
    for i in range(4, len(data)):
        if (data[i] - data[i-5]) < 20:
            class_of_peak = 1
            deltas.append(class_of_peak)
        elif (data[i] - data[i-5]) >= 20 and (data[i] - data[i-5]) < 40:
            class_of_peak = 2
            deltas.append(class_of_peak)
        elif (data[i] - data[i - 5]) >= 40 and (data[i] - data[i - 5]) < 60:
            class_of_peak = 3
            deltas.append(class_of_peak)
        elif (data[i] - data[i - 5]) >= 60 and (data[i] - data[i - 5]) < 100:
            class_of_peak = 4
            deltas.append(class_of_peak)
        #elif (data[i] - data[i - 5]) >= 100 :
        else:
            class_of_peak = 5
            deltas.append(class_of_peak)
    class_of_peak=max(deltas)
    return(class_of_peak)




def main(dataset):
    df = createdataframe(dataset)
    #filtered_int, time = loop_per_row(df)
    #print(filtered_int, time)
    #int_data, time = loop_per_row(df)
    #print(int_data, time)

    '''tO GET CLASSES, USE CODE BELOW:'''
    #THIS WAS OLD INDIVIDUAL TEST
    #class_of_peak=loop_per_row(df)
    #print(class_of_peak)
    df_with_class=loop_per_row(df)
    print(df_with_class)
    decision_tree(df_with_class)


    #plt.plot(time, int_data)
    #plt.plot(time, filtered_int)
    #plt.show()
    #need to convert string to list of integers


#below works, but I can't get a good print out
#nlogo_mixed_list_to_dict_rec('belief-spread-exp-results.csv')
#make this work currently is nothing


#my code
main('belief-spread-exp-results.csv')

def decision_tree(df):
    df = df.drop(columns=["data"])
    y=df["class"]
    y.to_numpy()
    df = df.reset_index(drop=True)
    water_feature_names= ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence', 'flint-community-size']
    X=df.to_numpy()
    class_names= ['1','2','3','4','5']
    print(X)
    print(y)
    y.to_numpy()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("first test")
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names = water_feature_names, class_names = class_names, filled = True, rounded = True, special_characters = True)
    graph = graphviz.Source(dot_data)
    graph

#activate_nicks_code('belief-spread-exp-results.csv')


#need to delete '[' etc- should take another peak at Nick's code to see if i can get this to work- his parsing should be super helpful we just need to understand!!