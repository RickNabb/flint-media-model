import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#running this for
df = pd.read_csv('media-connections-dynamic-organizing-exp-results_dtw.csv')
#df.columns = ['run','n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'organizing-capacity', 'organizing-strategy','repetition','data', 'dtw']
print(df)

plt.figure(figsize=(8,6))
plt.hist(df["dtw"], bins=100, alpha=0.5, label="data1")
plt.xlabel("DTW", size=14)
plt.ylabel("Count", size=14)
plt.title("Histogram of DTW from Media Connections Dynamic Organizing")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df["rsme"], bins=100, alpha=0.5, label="data1")
plt.xlabel("RMSE", size=14)
plt.ylabel("Count", size=14)
plt.title("Histogram of RMSE from Media Connections Dynamic Organizing")
plt.show()

sns.boxplot(data=df, x='organizing-strategy', y='rsme', hue='organizing-capacity')
    #sns.scatterplot(data=df_with_class, x='class-height', y='class-time')
plt.tick_params(axis='both', which='major', labelsize=6)
plt.xlabel('Organizing capacity')
plt.ylabel('RMSE')
plt.title('Citizen-citizen-influence and simple spread chance on RMSE')
plt.legend(loc='upper right')
plt.show()


df['label']=None
for i in range(0, df.shape[0]):
    #simple spread: 0.01, 0.05
    #ba-m: 3, 10
    #orgcap: 1,5,10
    #strat: neighb-of-n, high degree media, high degree cit, high degree cit and negih
    if df['simple-spread-chance'].iloc[i] == 0.01:
        if df['ba-m'].iloc[i] == 3:
            if df["organizing-capacity"].iloc[i] ==1:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 1
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 2
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 3
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 4
            elif df["organizing-capacity"].iloc[i] ==5:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 5
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 6
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 7
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 8
            else: #org-cap=10
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 9
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 10
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 11
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 12
        else: #ba-m = 10
            if df["organizing-capacity"].iloc[i] ==1:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 13
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 14
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 15
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 16
            elif df["organizing-capacity"].iloc[i] ==5:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 17
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 18
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 19
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 20
            else: #org-cap=10
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 21
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 22
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 23
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 24
    else: #simple spread chance=0.05
        if df['ba-m'].iloc[i] == 3:
            if df["organizing-capacity"].iloc[i] ==1:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 25
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 26
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 27
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 28
            elif df["organizing-capacity"].iloc[i] ==5:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 29
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 30
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 31
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 32
            else: #org-cap=10
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 33
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 34
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 35
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 36
        else: #ba-m = 10
            if df["organizing-capacity"].iloc[i] ==1:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 37
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 38
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 39
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 40
            elif df["organizing-capacity"].iloc[i] ==5:
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 41
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 42
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 43
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 44
            else: #org-cap=10
                if df['organizing-strategy'].iloc[i]== "neighbors-of-neighbors":
                    df.iat[i, 15] = 45
                elif df['organizing-strategy'].iloc[i]== "high-degree-media":
                    df.iat[i, 15] = 46
                elif df['organizing-strategy'].iloc[i]== "high-degree-citizens":
                    df.iat[i, 15] = 47
                else: #high-degree-cit-and-media
                    df.iat[i, 15] = 48
#df.to_csv("with_labels.csv")

print(df)
sns.scatterplot(data=df, x='label', y='rsme')
plt.title("RSME of scenarios")
plt.show()





