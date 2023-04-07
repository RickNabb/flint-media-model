import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_infl_mod_sweep_above_thresh=pd.read_csv("infl-model-sweep-eval.csv")
df_infl_mod_sweep_above_thresh['model']='infl_mod_sweep'
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
print(df_infl_mod_sweep_above_thresh)
df_dyn_org_above_thresh=pd.read_csv("dyn-org-sweep-eval.csv")
df_dyn_org_above_thresh['model']='dyn_org_sweep'
df_dyn_org_above_thresh=df_dyn_org_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
df_med_conn_dyn_org_above_thresh=pd.read_csv("med-con-dyn-org-sweep-analysis.csv")
df_med_conn_dyn_org_above_thresh['model']='med-con-dyn-org'
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
df_med_conn_inf_above_thresh=pd.read_csv("med-con-infl-analysis.csv")
df_med_conn_inf_above_thresh['model']='med-con-infl'
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])

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

sns.boxplot(data=df_metrics, y='rmse', x='model')
plt.title("RMSE Distribution for Each Model")
plt.legend()
plt.ylabel("RMSE")
plt.xlabel('Model')
plt.show()

sns.boxplot(data=df_metrics, y='total_error', x='model')
plt.title("Total Error Distribution for Each Model")
plt.legend()
plt.ylabel("Total Error")
plt.xlabel('Model')
plt.show()




#will show 20 smallest rmse
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.nsmallest(50, 'rmse')
df_dyn_org_above_thresh=df_dyn_org_above_thresh.nsmallest(50, 'rmse')
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.nsmallest(50, 'rmse')
df_med_conn_inf_above_thresh.to_csv("med_con_inf_best_50_rsme.csv")
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.nsmallest(50, 'rmse')
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

#will show 20 smallest PEARSONS R
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.nlargest(50, 'corr_coef')
df_dyn_org_above_thresh=df_dyn_org_above_thresh.nlargest(50, 'corr_coef')
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.nlargest(50, 'corr_coef')
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.nlargest(50, 'corr_coef')
df_metrics_smallest_cc=df_med_conn_inf_above_thresh.append(df_med_conn_dyn_org_above_thresh)
df_metrics_smallest_cc=df_metrics_smallest_cc.append(df_dyn_org_above_thresh)
df_metrics_smallest_cc=df_metrics_smallest_cc.append(df_infl_mod_sweep_above_thresh)

print(df_metrics_smallest_cc)

sns.boxplot(data=df_metrics_smallest_cc, y='corr_coef', x='model')
plt.title("50 Largest Pearson's R Per Model")
plt.legend()
plt.ylabel("Pearson's R")
plt.xlabel('Model')
plt.show()


df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.nsmallest(50, 'total_error')
df_dyn_org_above_thresh=df_dyn_org_above_thresh.nsmallest(50, 'total_error')
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.nsmallest(50, 'total_error')
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.nsmallest(50, 'total_error')
df_metrics_smallest_rmse=df_med_conn_inf_above_thresh.append(df_med_conn_dyn_org_above_thresh)
df_metrics_smallest_rmse=df_metrics_smallest_rmse.append(df_dyn_org_above_thresh)
df_metrics_smallest_rmse=df_metrics_smallest_rmse.append(df_infl_mod_sweep_above_thresh)

sns.boxplot(data=df_metrics_smallest_rmse, y='total_error', x='model')
plt.title("50 Lowest Total Errors Per Model")
plt.legend()
plt.ylabel("Total Errors")
plt.xlabel('Model')
plt.show()

#attempt 3 d plot
plt.subplots(figsize=(8,8))
sns.scatterplot(data=df_metrics, y='rmse', x='corr_coef', hue='model')
plt.title("RMSE vs Pearson's R for all 4 models")
plt.xlabel("Pearson's R")
plt.ylabel("RMSE")
plt.legend()
plt.show()


#make a dataframe (and csv of 50 best runs based on each metric of influence model)
df_infl_mod_sweep_above_thresh=pd.read_csv("infl-model-sweep-eval.csv")
df_infl_mod_sweep_above_thresh["metric"]=None
df_infl_mod_sweep_above_thresh_total_error=df_infl_mod_sweep_above_thresh.nsmallest(50, 'total_error')
df_infl_mod_sweep_above_thresh_total_error["metric"]='total_error'
df_infl_mod_sweep_above_thresh_rmse=df_infl_mod_sweep_above_thresh.nsmallest(50, 'rmse')
df_infl_mod_sweep_above_thresh_rmse["metric"]='rmse'
df_infl_mod_sweep_above_thresh_r=df_infl_mod_sweep_above_thresh.nlargest(50, 'corr_coef')
df_infl_mod_sweep_above_thresh_r["metric"]='r'
df_metrics_best_fit=df_infl_mod_sweep_above_thresh_total_error.append(df_infl_mod_sweep_above_thresh_r)
df_metrics_best_fit=df_metrics_best_fit.append(df_infl_mod_sweep_above_thresh_rmse)
df_metrics_best_fit=df_metrics_best_fit.drop_duplicates(subset=['run'])

df_metrics_best_fit.to_csv("best_50_cases_per_metric_inf_sweep.csv")
print(df_metrics_best_fit)



#compare model fit 1 with adding organizng and media connections:
df_infl_mod_sweep_above_thresh=pd.read_csv("infl-model-sweep-eval.csv")
df_infl_mod_sweep_above_thresh['model']='infl_mod_sweep'
df_med_conn_inf_above_thresh=pd.read_csv("med-con-infl-analysis.csv")
df_med_conn_inf_above_thresh['model']='med-con-infl'
#now to specify model parameters of interest only\
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['ba-m']==10]
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['simple-spread-chance']==0.05]
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['citizen-citizen-influence']==0.75]
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['citizen-media-influence']==0.75]
df_infl_mod_sweep_above_thresh=df_infl_mod_sweep_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
print(df_infl_mod_sweep_above_thresh)
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['ba-m']==10]
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['simple-spread-chance']==0.05]
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['citizen-citizen-influence']==0.75]
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['citizen-media-influence']==0.75]
df_med_conn_inf_above_thresh=df_med_conn_inf_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
print(df_med_conn_inf_above_thresh)
#male a figure of all of these plots maybe? time series graph?

df_metrics_scen_1=df_med_conn_inf_above_thresh.append(df_infl_mod_sweep_above_thresh)
print(df_metrics_scen_1)

sns.boxplot(data=df_metrics_scen_1, x="model", y='rmse')
plt.title("RMSE for Scenario 1")
plt.xlabel("model")
plt.ylabel("RMSE")
plt.show()

sns.boxplot(data=df_metrics_scen_1, x="model", y='corr_coef')
plt.title("Pearson's R for Scenario 1")
plt.xlabel("model")
plt.ylabel("Pearson's R")
plt.show()

sns.boxplot(data=df_metrics_scen_1, x="model", y='total_error')
plt.title("Total error for Scenario 1")
plt.xlabel("model")
plt.ylabel("total error")
plt.show()


#what about multiscenario?

#infl_model_sweep
df_infl_mod_sweep_above_thresh=pd.read_csv("infl-model-sweep-eval.csv")
df_infl_mod_sweep_above_thresh['model']='infl_mod_sweep'

#now to specify model parameters of interest only\
scen1_infl=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['ba-m']==10]
scen1_infl=scen1_infl[scen1_infl['simple-spread-chance']==0.05]
scen1_infl=scen1_infl[scen1_infl['citizen-citizen-influence']==0.75]
scen1_infl=scen1_infl[scen1_infl['citizen-media-influence']==0.75]
scen1_infl=scen1_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen1_infl["scenario"]=1

df_med_conn_inf_above_thresh=pd.read_csv("med-con-infl-analysis.csv")
df_med_conn_inf_above_thresh['model']='med-con-infl'
scen1_med_infl=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['ba-m']==10]
scen1_med_infl=scen1_med_infl[scen1_med_infl['simple-spread-chance']==0.05]
scen1_med_infl=scen1_med_infl[scen1_med_infl['citizen-citizen-influence']==0.75]
scen1_med_infl=scen1_med_infl[scen1_med_infl['citizen-media-influence']==0.75]
scen1_med_infl=scen1_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen1_med_infl["scenario"]=1

scen2_infl=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['ba-m']==3]
scen2_infl=scen2_infl[scen2_infl['simple-spread-chance']==0.05]
scen2_infl=scen2_infl[scen2_infl['citizen-citizen-influence']==0.75]
scen2_infl=scen2_infl[scen2_infl['citizen-media-influence']==0.75]
scen2_infl=scen2_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen2_infl["scenario"]=2

scen2_med_infl=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['ba-m']==3]
scen2_med_infl=scen2_med_infl[scen2_med_infl['simple-spread-chance']==0.05]
scen2_med_infl=scen2_med_infl[scen2_med_infl['citizen-citizen-influence']==0.75]
scen2_med_infl=scen2_med_infl[scen2_med_infl['citizen-media-influence']==0.75]
scen2_med_infl=scen2_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen2_med_infl["scenario"]=2

scen3_infl=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['ba-m']==10]
scen3_infl=scen3_infl[scen3_infl['simple-spread-chance']==0.01]
scen3_infl=scen3_infl[scen3_infl['citizen-citizen-influence']==0.75]
scen3_infl=scen3_infl[scen3_infl['citizen-media-influence']==0.75]
scen3_infl=scen3_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen3_infl["scenario"]=3

scen3_med_infl=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['ba-m']==10]
scen3_med_infl=scen3_med_infl[scen3_med_infl['simple-spread-chance']==0.01]
scen3_med_infl=scen3_med_infl[scen3_med_infl['citizen-citizen-influence']==0.75]
scen3_med_infl=scen3_med_infl[scen3_med_infl['citizen-media-influence']==0.75]
scen3_med_infl=scen3_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen3_med_infl["scenario"]=3

scen4_infl=df_infl_mod_sweep_above_thresh[df_infl_mod_sweep_above_thresh['ba-m']==3]
scen4_infl=scen4_infl[scen4_infl['simple-spread-chance']==0.01]
scen4_infl=scen4_infl[scen4_infl['citizen-citizen-influence']==0.75]
scen4_infl=scen4_infl[scen4_infl['citizen-media-influence']==0.75]
scen4_infl=scen4_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen4_infl["scenario"]=4

scen4_med_infl=df_med_conn_inf_above_thresh[df_med_conn_inf_above_thresh['ba-m']==3]
scen4_med_infl=scen4_med_infl[scen4_med_infl['simple-spread-chance']==0.01]
scen4_med_infl=scen4_med_infl[scen4_med_infl['citizen-citizen-influence']==0.75]
scen4_med_infl=scen4_med_infl[scen4_med_infl['citizen-media-influence']==0.75]
scen4_med_infl=scen4_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-citizen-influence', 'citizen-media-influence', 'flint-community-size', 'repetition', 'data', 'short list', 'threshold'])
scen4_med_infl["scenario"]=4


df_scen_comp=scen1_infl.append(scen1_med_infl)
df_scen_comp=df_scen_comp.append(scen2_infl)
df_scen_comp=df_scen_comp.append(scen2_med_infl)
df_scen_comp=df_scen_comp.append(scen3_infl)
df_scen_comp=df_scen_comp.append(scen3_med_infl)
df_scen_comp=df_scen_comp.append(scen4_infl)
df_scen_comp=df_scen_comp.append(scen4_med_infl)

sns.boxplot(data=df_scen_comp, x="scenario", y="rmse", hue='model')
plt.title("Scenario Comparison Adding Media Connections and Organizing")
plt.xlabel("Scenario")
plt.ylabel('RMSE')
plt.legend()
plt.show()

sns.boxplot(data=df_scen_comp, x="scenario", y="corr_coef", hue='model')
plt.title("Scenario Comparison Adding Media Connections and Organizing")
plt.xlabel("Scenario")
plt.ylabel("Pearson's R")
plt.legend()
plt.show()

sns.boxplot(data=df_scen_comp, x="scenario", y="total_error", hue='model')
plt.title("Scenario Comparison Adding Media Connections and Organizing")
plt.xlabel("Scenario")
plt.ylabel("Total Error")
plt.legend()
plt.show()

'''COMPARING MODELS 2 AND 4'''

df_dyn_org_above_thresh=pd.read_csv("dyn-org-sweep-eval.csv")
df_dyn_org_above_thresh['model']='dyn_org_sweep'
df_dyn_org_above_thresh=df_dyn_org_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
df_med_conn_dyn_org_above_thresh=pd.read_csv("med-con-dyn-org-sweep-analysis.csv")
df_med_conn_dyn_org_above_thresh['model']='med-con-dyn-org'
df_med_conn_dyn_org_above_thresh=df_med_conn_dyn_org_above_thresh.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])


#DYN_ORG_model_sweep
df_dyn_org_above_thresh=pd.read_csv("dyn-org-sweep-eval.csv")
df_dyn_org_above_thresh['model']='dyn_org_sweep'

#now to specify model parameters of interest only\
scen1_infl=df_dyn_org_above_thresh[df_dyn_org_above_thresh['ba-m']==10]
scen1_infl=scen1_infl[scen1_infl['simple-spread-chance']==0.05]
scen1_infl=scen1_infl[scen1_infl['organizing-capacity']==5]
scen1_infl=scen1_infl[scen1_infl['organizing-strategy']=='high-degree-media']
scen1_infl=scen1_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen1_infl["scenario"]=1

df_med_conn_dyn_org_above_thresh=pd.read_csv("med-con-dyn-org-sweep-analysis.csv")
df_med_conn_dyn_org_above_thresh['model']='med-con-dyn-org'
scen1_med_infl=df_med_conn_dyn_org_above_thresh[df_med_conn_dyn_org_above_thresh['ba-m']==10]
scen1_med_infl=scen1_med_infl[scen1_med_infl['simple-spread-chance']==0.05]
scen1_med_infl=scen1_med_infl[scen1_med_infl['organizing-capacity']==5]
scen1_med_infl=scen1_med_infl[scen1_med_infl['organizing-strategy']=='high-degree-media']
scen1_med_infl=scen1_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen1_med_infl["scenario"]=1

scen2_infl=df_dyn_org_above_thresh[df_dyn_org_above_thresh['ba-m']==3]
scen2_infl=scen2_infl[scen2_infl['simple-spread-chance']==0.05]
scen2_infl=scen2_infl[scen2_infl['organizing-capacity']==5]
scen2_infl=scen2_infl[scen2_infl['organizing-strategy']=='high-degree-media']
scen2_infl=scen2_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen2_infl["scenario"]=2

scen2_med_infl=df_med_conn_dyn_org_above_thresh[df_med_conn_dyn_org_above_thresh['ba-m']==3]
scen2_med_infl=scen2_med_infl[scen2_med_infl['simple-spread-chance']==0.05]
scen2_med_infl=scen2_med_infl[scen2_med_infl['organizing-capacity']==5]
scen2_med_infl=scen2_med_infl[scen2_med_infl['organizing-strategy']=='high-degree-media']
scen2_med_infl=scen2_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen2_med_infl["scenario"]=2


scen3_infl=df_med_conn_dyn_org_above_thresh[df_med_conn_dyn_org_above_thresh['ba-m']==10]
scen3_infl=scen3_infl[scen3_infl['simple-spread-chance']==0.05]
scen3_infl=scen3_infl[scen3_infl['organizing-capacity']==5]
scen3_infl=scen3_infl[scen3_infl['organizing-strategy']=='neighbors-of-neighbors']
scen3_infl=scen3_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen3_infl["scenario"]=3

scen3_med_infl=df_dyn_org_above_thresh[df_dyn_org_above_thresh['ba-m']==10]
scen3_med_infl=scen3_med_infl[scen3_med_infl['simple-spread-chance']==0.05]
scen3_med_infl=scen3_med_infl[scen3_med_infl['organizing-capacity']==5]
scen3_med_infl=scen3_med_infl[scen3_med_infl['organizing-strategy']=='neighbors-of-neighbors']
scen3_med_infl=scen3_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen3_med_infl["scenario"]=3

scen4_infl=df_med_conn_dyn_org_above_thresh[df_med_conn_dyn_org_above_thresh['ba-m']==10]
scen4_infl=scen4_infl[scen4_infl['simple-spread-chance']==0.05]
scen4_infl=scen4_infl[scen4_infl['organizing-capacity']==10]
scen4_infl=scen4_infl[scen4_infl['organizing-strategy']=='neighbors-of-neighbors']
scen4_infl=scen4_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen4_infl["scenario"]=4

scen4_med_infl=df_dyn_org_above_thresh[df_dyn_org_above_thresh['ba-m']==10]
scen4_med_infl=scen4_med_infl[scen4_med_infl['simple-spread-chance']==0.05]
scen4_med_infl=scen4_med_infl[scen4_med_infl['organizing-capacity']==10]
scen4_med_infl=scen4_med_infl[scen4_med_infl['organizing-strategy']=='neighbors-of-neighbors']
scen4_med_infl=scen4_med_infl.drop(columns=['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m','organizing-capacity', 'organizing-strategy', 'repetition', 'data', 'short list', 'threshold'])
scen4_med_infl["scenario"]=4


df_scen_comp=scen1_infl.append(scen1_med_infl)
df_scen_comp=df_scen_comp.append(scen2_infl)
df_scen_comp=df_scen_comp.append(scen2_med_infl)
df_scen_comp=df_scen_comp.append(scen3_infl)
df_scen_comp=df_scen_comp.append(scen3_med_infl)
df_scen_comp=df_scen_comp.append(scen4_infl)
df_scen_comp=df_scen_comp.append(scen4_med_infl)

sns.boxplot(data=df_scen_comp, x="scenario", y="rmse", hue='model')
plt.title("Scenario Comparison Model 2 + 4 (Media)")
plt.xlabel("Scenario")
plt.ylabel('RMSE')
plt.legend()
plt.show()

sns.boxplot(data=df_scen_comp, x="scenario", y="corr_coef", hue='model')
plt.title("Scenario Comparison Model 2 + 4 (Media)")
plt.xlabel("Scenario")
plt.ylabel("Pearson's R")
plt.legend()
plt.show()

sns.boxplot(data=df_scen_comp, x="scenario", y="total_error", hue='model')
plt.title("Scenario Comparison Model 2 + 4 (Media)")
plt.xlabel("Scenario")
plt.ylabel("Total Error")
plt.legend()
plt.show()

'''QUICK FIGURE ON ORG STRATEGY'''

df_dyn_org_above_thresh=pd.read_csv("dyn-org-sweep-eval.csv")
df_dyn_org_above_thresh["model"]='2 (No Media)'

df_med_conn_dyn_org_above_thresh=pd.read_csv("med-con-dyn-org-sweep-analysis.csv")
df_med_conn_dyn_org_above_thresh['model']='4 (Media)'
df_org_strat=df_dyn_org_above_thresh.append(df_med_conn_dyn_org_above_thresh)

sns.boxplot(data=df_dyn_org_above_thresh, x="organizing-strategy", y='rmse', hue='model')
plt.title("Model 2 Results Across Organizing Strategy")
plt.xlabel("Organizing Strategy")
plt.ylabel("RMSE")
plt.show()

sns.boxplot(data=df_dyn_org_above_thresh, x="organizing-strategy", y='corr_coef', hue='model')
plt.title("Model 2 Results Across Organizing Strategy")
plt.xlabel("Organizing Strategy")
plt.ylabel("Pearson's R")
plt.show()

sns.boxplot(data=df_dyn_org_above_thresh, x="organizing-strategy", y='total_error', hue='model')
plt.title("Model 2 Results Across Organizing Strategy")
plt.xlabel("Organizing Strategy")
plt.ylabel("Total Error")
plt.show()