from sklearn import metrics
#import fsi_analysis4 as fs4 # takes time since it's doing a bootstrap process
import pandas
import numpy as np

fname = 'bootstrap_pred_results_k1_L1_2026-Mar-27-14:08.csv'

df = pandas.read_csv(fname)
nb = len(df['bootstrap_number'].unique())

doh = np.zeros((nb,4))
for i,dfs in df.groupby('bootstrap_number'):
    doh[i,0] = metrics.accuracy_score(dfs['true_label'],dfs['pred_lr'])
    doh[i,1] = metrics.f1_score(dfs['true_label'],dfs['pred_lr'])
    doh[i,2] = metrics.accuracy_score(dfs['true_label'],dfs['pred_rf'])
    doh[i,3] = metrics.f1_score(dfs['true_label'],dfs['pred_rf'])
    if i%(nb//10)==0:
        print(f'{i} of {nb}')
#
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
Q = np.quantile(doh, quantiles, axis=0)

df_bootstraps_summary = pandas.DataFrame(Q, columns=['LR_acc','LR_f1', 'RF_acc', 'RF_f1'])
df_bootstraps_summary['quantile'] = quantiles

_z = fname.split('.')
df_bootstraps_summary.to_csv(f'{_z[0]}_SUMMARY.{_z[1]}', index=None)
