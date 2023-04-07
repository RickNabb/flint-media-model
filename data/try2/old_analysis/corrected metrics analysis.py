import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#no threshold cutoff here-only on comp desktop
df_infl_mod_sweep_above_thresh=pd.read_csv("infl-model-sweep-eval.csv")
df_infl_mod_sweep_above_thresh['model']='infl_mod_sweep'
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
print(df_infl_mod_sweep_above_thresh)
df_dyn_org_above_thresh=pd.read_csv("dyn-org-sweep-eval.csv")
df_dyn_org_above_thresh['model']='dyn_org_sweep'
df_dyn_org_above_thresh=df_dyn_org_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
df_med_conn_dyn_org_above_thresh=pd.read_csv("media-con-dyn-org-eval.csv")
df_med_conn_dyn_org_above_thresh['model']='med-con-dyn-org'
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
df_med_conn_inf_above_thresh=pd.read_csv("media-con-infl-sweep-eval.csv")
df_med_conn_inf_above_thresh['model']='med-con-infl'
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])

#print data on metrics
print('infl model sweep')
print(df_infl_mod_sweep_above_thresh['rmse'].mean())



df_metrics=df_med_conn_inf_above_thresh.append(df_med_conn_dyn_org_above_thresh)
df_metrics=df_metrics.append(df_dyn_org_above_thresh)
df_metrics=df_metrics.append(df_infl_mod_sweep_above_thresh)

print(df_metrics)

sns.boxplot(data=df_metrics, y='corr_coef', x='model')
plt.title("Pearson's R Distribution for Each Model")
plt.legend()
plt.ylabel("Pearson's R")
plt.xlabel('Model')
plt.show()


#will show 20 smallest rmse
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.nsmallest(20, 'rmse')
df_dyn_org_above_thresh=df_dyn_org_above_thresh.nsmallest(20, 'rmse')
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.nsmallest(20, 'rmse')
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.nsmallest(20, 'rmse')
df_metrics_smallest_rmse=df_med_conn_inf_above_thresh.append(df_med_conn_dyn_org_above_thresh)
df_metrics_smallest_rmse=df_metrics_smallest_rmse.append(df_dyn_org_above_thresh)
df_metrics_smallest_rmse=df_metrics_smallest_rmse.append(df_infl_mod_sweep_above_thresh)

print(df_metrics_smallest_rmse)

sns.boxplot(data=df_metrics_smallest_rmse, y='rmse', x='model')
plt.title("50 Lowest RMSE Per Model")
plt.legend()
plt.ylabel("RMSE")
plt.xlabel('Model')
plt.show()

#will show 20 largest PEARSONS R
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.nlargest(50, 'corr_coef')
df_dyn_org_above_thresh=df_dyn_org_above_thresh.nlargest(50, 'corr_coef')
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.nlargest(50, 'corr_coef')
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.nlargest(50, 'corr_coef')
df_metrics_smallest_cc=df_med_conn_inf_above_thresh.append(df_med_conn_dyn_org_above_thresh)
df_metrics_smallest_cc=df_metrics_smallest_cc.append(df_dyn_org_above_thresh)
df_metrics_smallest_cc=df_metrics_smallest_cc.append(df_infl_mod_sweep_above_thresh)

print(df_metrics_smallest_cc)

sns.boxplot(data=df_metrics_smallest_cc, y='corr_coef', x='model')
plt.title("50 Lowest Pearson's R Per Model")
plt.legend()
plt.ylabel("Pearson's R")
plt.xlabel('Model')
plt.show()