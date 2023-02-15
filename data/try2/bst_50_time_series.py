import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("best_50_cases_per_metric_inf_sweep.csv")
print(df)
df_for_seaborn=pd.DataFrame(columns=['time', 'new_agents', 'run', 'rmse', 'total_error', 'corr_coef', 'metric_select'])
print(df_for_seaborn)

df_sea_media_conn_inf=pd.read_csv("df_sea_media_conn_ifl.csv")
#df_sea_media_conn_inf=df_sea_media_conn_inf.head(2280)



google_data=pd.read_csv("google-trends.csv")
google_data.columns = ['week', 'data']
g_data = google_data['data'].to_list()
google_data_format=pd.DataFrame(columns=['time', 'sim_num', 'simple-spread-chance', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence', 'new-agents', 'new-agents-scaled', 'rmse', 'corr_coef','total_error'], index=range(0,114))
print(g_data)
for i in range(0, len(g_data)):
    time=i
    google_data_format["time"].iloc[i]=time
    google_data_format['sim_num'].iloc[i]='google-trend'
    val=g_data[i]
    google_data_format['new-agents-scaled'].iloc[i]=val
df_sea_media_conn_inf=df_sea_media_conn_inf.append(google_data_format, ignore_index=True)
df_sea_media_conn_inf['mod-sim']=None
print(df_sea_media_conn_inf)
for i in range(0, df_sea_media_conn_inf.shape[0]):
    if df_sea_media_conn_inf["sim_num"].iloc[i]=='google-trend':
        df_sea_media_conn_inf["mod-sim"].iloc[i]='google'
    else:
        df_sea_media_conn_inf['mod-sim'].iloc[i]= 'simulation'
    #check this- not sure if dataframe is being written over

sns.lineplot(data=df_sea_media_conn_inf, x='time', y='new-agents-scaled', hue='mod-sim')
plt.show()