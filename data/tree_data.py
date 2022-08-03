import csv
import pandas as pd

def createdataframe(dataset):
    df =  pd.read_csv(dataset, sep='|' , engine='python')
    df.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence', 'flint-community-size', 'data']
    df.set_index('run', inplace=True)
    return df
    #to-do:
    #use index as first value
    #add column names
def printdataframe(dataset):
    x = createdataframe(dataset)
    print(x)

printdataframe('belief-spread-exp-results.csv')