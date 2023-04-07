import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_inf_mod_sweep=pd.read_csv("infl-model-sweep-eval.csv")
df_thresh=df_inf_mod_sweep[df_inf_mod_sweep["threshold"]==1]
print(df_thresh)

