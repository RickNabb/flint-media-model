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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols



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
    #for standard
    #df =  pd.read_csv(dataset, sep='|' , engine='python')
    #for gradual
    df = pd.read_csv(dataset)
    #use below for standard
    #df.columns = ['run','n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'organizing-capacity', 'organizing-strategy', 'repetition','data']

    #use below for gradual scalar
    df.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-media-influence','citizen-citizen-influence', 'flint-community-size', 'repetition','data']
    #print(df['simple-spread-chance'])
    #print(df['spread-type'])
    #df.drop(['spread-type'], axis=1)
    df_new=df.drop(columns=['spread-type', 'graph-type', 'n','run', 'repetition'])
    #print('after drop', df_new['spread-type'])
    #df_new.set_index('run', inplace=True)
    print(df_new)
    return df_new

def convertdata(data):
    #print(data)
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

def convert_to_float(data):
    for i in range(0, len(data)):
        data[i] = float(data[i])
    return data

def apply_filter(int_data):
    #newdata = savgol_filter(int_data, 15, 2)
    newdata = savgol_filter(int_data, 15, 2)
    return newdata
    #could define spike as x% of population in x time steps or use filter and check peak

def loop_per_row(df):
    df.insert(0, "class", " ")
    #print(df['simple-spread-chance'])
    for i in range(0,df.shape[0]):
        run1 = df.iloc[i]
        data = run1['data']
        finallist = convertdata(data)
        int_data = convert_to_int(finallist)
        #print(int_data)
        '''for full dataframe'''
        #for cart use below
        #class_of_peak = evaluate_peak(int_data)

        #for lr use below
        class_of_peak_time = evaluate_peak_time(int_data)
        df.at[i,'class-time'] = class_of_peak_time
        class_of_peak_height = evaluate_peak_lr(int_data)
        df.at[i, 'class-height'] = class_of_peak_height
        df['simple-spread-chance'] = df['simple-spread-chance'].astype(float)
        df['simple-spread-chance'] = df['simple-spread-chance'].astype(float)
        df['ba-m'] = df['ba-m'].astype(float)
        #df['organizing-capacity'] = df['organizing-capacity'].astype(float)
        #df['organizing-strategy']= df['organizing-strategy'].astype(float)
        #df['citizen-media-influence'] = df['citizen-media-influence'].astype(float)
        #df['citizen-citizen-influence'] = df['citizen-citizen-influence'].astype(float)
        #df['flint-community-size'] = df['flint-community-size'].astype(float)

        #INCLUDE WHEN USING GRADUAL SCALAR
        #df['citizen-media-gradual-scalar'] = df['citizen-media-gradual-scalar'].astype(float)
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

def evaluate_peak_lr(data):
    deltas=[]
    #captures with some idea of "time"
    #for i in range(4, len(data)):
    #    delta=(data[i] - data[i - 5])
    #    deltas.append(delta)
    #class_of_peak=max(deltas)
    #return(class_of_peak)
    #to just capture tallest height
    for i in range(0, len(data)):
        delta=(data[i])
        deltas.append(delta)
    class_of_peak=max(deltas)
    return(class_of_peak)


def evaluate_peak_time(data):
    #to set time it would be the loc of max peak
    #note: this would only capture later peak if two or more equivalent peaks
    deltas=[]
    maxval = max(data)
    #print(maxval)
    for i in range(0, len(data)):
        if data[i] == maxval:
            time_of_peak = i
        else:
            pass
    # Note: use following line if doing linear reg
    class_of_peak=time_of_peak
    return class_of_peak


def decision_tree(df):
    df1 = df.drop(columns=["data"])
    df1['class']=df1['class'].astype(float)
    y=df1["class"]
    y.to_numpy()
    df1=df1.drop(columns=['class'])
    df1 = df1.reset_index(drop=True)
    #water_feature_names= ['simple-spread-chance', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence', 'flint-community-size']
    water_feature_names = ['simple-spread-chance', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence', 'citizen-media-gradual-scalar','flint-community-size']
    X=df1.to_numpy()
    class_names= ['1','2','3','4','5']
    print(X)
    print(y)
    y.to_numpy()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    #dot_data = tree.export_graphviz(clf, out_file=None, feature_names=water_feature_names, class_names=class_names, filled)
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=water_feature_names, class_names=class_names,
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("gradual_height")
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names = water_feature_names, class_names = class_names, filled = True, rounded = True, special_characters = True)
    #graph = graphviz.Source(dot_data)
    ###may need to convert other categories to floats- needs more work####

def linear_reg(df_with_class):
    df1 = df_with_class.drop(columns=["data"])
    df1['class'] = df1['class'].astype(float)
    #use below for initial results
    X=df1[['simple-spread-chance', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence']]
    #use below for gradual scalar
    #X = df1[['simple-spread-chance', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence','citizen-media-gradual-scalar','flint-community-size']]
    y = df1["class"]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    # with statsmodels
    x = sm.add_constant(X)  # adding a constant

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)

    print_model = model.summary()
    print(print_model)

    #TO-DO
        #CHECK RESIDUALS
        #CHECK ASSUMPTIONS

'''MAIN CODE FOR DECISION TREE'''
def main_cart_belief(dataset):
    df_adj = createdataframe(dataset)
    '''tO GET CLASSES, USE CODE BELOW:'''
    df_with_class=loop_per_row(df_adj)
    #(df_with_class)
    decision_tree(df_with_class)

def main_linearreg_belief(dataset):
    df_adj = createdataframe(dataset)
    df_with_class = loop_per_row(df_adj)
    linear_reg(df_with_class)

def make_histograms(dataset):
    #plt 1: scatterplot of all results
    df_adj = createdataframe(dataset)
    df_with_class = loop_per_row(df_adj)
    sns.scatterplot(data=df_with_class, x='class-height', y='class-time')
    plt.xlabel('Height of Peak')
    plt.ylabel('Time of Max Peak')
    plt.title('Height vs Time of Peak for Parameter Sweep')
    plt.show()

    #plot 2: simple-spread-chance height
    sns.boxplot(data=df_with_class, x='simple-spread-chance', y='class-height')
    plt.xlabel('Simple Spread Chance')
    plt.ylabel('Height of Max Peak')
    plt.title('Impact of Simple Spread Chance on Peak Height')
    plt.show()

    #plot 3: simple-spread-chance time
    sns.boxplot(data=df_with_class, x='simple-spread-chance', y='class-time')
    plt.xlabel('Simple Spread Chance')
    plt.ylabel('Time of Max Peak')
    plt.title('Impact of Simple Spread Chance on Time of Peak')
    plt.show()

    # plot 3: simple-spread-chance time
    sns.boxplot(data=df_with_class, x='simple-spread-chance', y='class-time')
    plt.xlabel('Simple Spread Chance')
    plt.ylabel('Time of Max Peak')
    plt.title('Impact of Simple Spread Chance on Time of Peak')
    plt.show()

    # plot 3: ba-m
    sns.boxplot(data=df_with_class, x='ba-m', y='class-height')
    plt.xlabel('ba-m')
    plt.ylabel('Height of Max Peak')
    plt.title('Impact of ba-m on height of Peak')
    plt.show()

    #plot 4: ba-m time
    sns.boxplot(data=df_with_class, x='ba-m', y='class-time')
    plt.xlabel('ba-m')
    plt.ylabel('Time of Max Peak')
    plt.title('Impact of ba-m on time of Peak')
    plt.show()

    # plot 5: citizen-media-influence time
    sns.boxplot(data=df_with_class, x='citizen-media-influence', y='class-time')
    plt.xlabel('citizen-media-influence')
    plt.ylabel('Time of Max Peak')
    plt.title('Impact of citizen-media influence on time of Peak')
    plt.show()

    #plot 6: citizen-media-influence height
    sns.boxplot(data=df_with_class, x='citizen-media-influence', y='class-height')
    plt.xlabel('citizen-media-influence')
    plt.ylabel('Height of Max Peak')
    plt.title('Impact of Citizen-Media Influence on height of Peak')
    plt.show()

    #plt 7: citizen-citizen-influence height
    sns.boxplot(data=df_with_class, x='citizen-citizen-influence', y='class-height')
    plt.xlabel('citizen-citizen-influence')
    plt.ylabel('Height of Max Peak')
    plt.title('Impact of Citizen-Citizen Influence on height of Peak')
    plt.show()

    # plt 8: citizen-citizen-influence time
    sns.boxplot(data=df_with_class, x='citizen-citizen-influence', y='class-time')
    plt.xlabel('citizen-citizen-influence')
    plt.ylabel('Time of Max Peak')
    plt.title('Impact of Citizen-Citizen Influence on time of Peak')
    plt.show()


make_histograms('media-connections-influence-model-sweep.csv')