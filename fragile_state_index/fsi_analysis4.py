from matplotlib import pyplot as plt
from matplotlib import ticker
import pandas
import numpy as np

import load_fsi
#import vis_tools_fsi
import load_tmk
import datasets
import seaborn

#

import datetime
tstamp = datetime.datetime.now().strftime('%Y-%b-%d-%H:%M')

from sklearn import linear_model # for Lasso, etc.
from sklearn import metrics      # for, e.g., sklearn.metrics.roc_auc_score

########
'''
RESEARCH QUESTION:
    
    Given the FSI data from k years; year_{i}, year_{i-1}, ..., year_{i-k+1}, 
    predict the likelihood of an event at year_{i+L}.
    
    k : length of observation/memory
    L : forecast length (start with L=1).
    
    There are 12 indicators; so we are mapping 12k dimensions to 1 
    dimension. 
    
    In adjustment to analysis3: I would like to add two 
    nuances to investigate the prediction problem:
        1. Select only TMK events after a period of peace (k years), 
           chronologically, for a given country; the rest must be thrown out.
           (expecting fewer positive examples)
        2. Do some type of train/test methodology, 
           even with this methodology.
        3. (For the future): compare countries' data points
           restricted to a given year. For example, 
           build a classifier based on data year 2010.
'''

# Targeted Mass Killings data since 2006
k=1
L=1
X,y,meta = datasets.build_fsi_predicting_tmk(k=k, L=L, track_ongoing=False)

features=meta['features']

# final labeled data set (!!)

###############

# naive fit produces a null model with Lasso.
# handle the imbalanced classes by 
# repeated training of all TMK cases (59 of them?)
# versus an 59 uniform iid selected non-TMK.

not_tmk_idx = np.where(y==0)[0]
yes_tmk_idx = np.where(y>0)[0]
ntmk = len(yes_tmk_idx)
#ntmk = ntmk//2 # allow for random sampling of the tmk events.
print("k: %i, L: %i, ntmk: %i"%(k,L,ntmk) )

np.random.seed(10072023)
nboots = 10000
models = []
models_coef_ = np.zeros( (nboots, 12*k) )

subsets = np.zeros((nboots, 2*ntmk), dtype=int)
y_trues = np.zeros((nboots, 2*ntmk))
y_preds = np.zeros((nboots, 2*ntmk))
aucrocs = np.zeros(nboots)

for i in range(nboots):
    model = linear_model.ElasticNet(max_iter=1000, l1_ratio=0.05, positive=False) # idk lol
    
    subset = np.concatenate([
        np.random.choice(yes_tmk_idx, ntmk, replace=False), 
        np.random.choice(not_tmk_idx, ntmk, replace=False)
    ])
    
    subsets[i] = subset
    
    model.fit(X[subset], y[subset])
    ypred = model.predict(X[subset])
    
    y_trues[i] = y[subset]
    y_preds[i] = ypred
    # can do more sophisticated things later...
    try:
        aucrocs[i] = metrics.roc_auc_score(y_trues[subset], ypred)
    except:
        # TODO: think through.
        # probably doing regression instead of classification
        pass
    
    models.append(model)
    models_coef_[i] = model.coef_
    
    if i%(nboots//10)==0:
        print(f'{i+1} of {nboots}')
#

df_weights = pandas.DataFrame(data=models_coef_,columns=features)

# build long dataframe solely for the purposes of visualization
# (seaborn aggregates/does errorbars this way; can also color bars by indicator group)
df_results = df_weights.melt(var_name='Indicator', value_name='Coefficient')
df_results['Indicator group'] = [{'X':'S'}.get(v[0],v[0]) for v in df_results['Indicator']]

if __name__=="__main__":
    plt.rcParams.update({'legend.framealpha': 1})

    fig,ax = plt.subplots(figsize=(12,8), constrained_layout=True)
    seaborn.set_context("paper", font_scale=2) # font_scale doesn't seem to do anything.
    seaborn.set_style("whitegrid")
    
    seaborn.barplot(data=df_results, y='Indicator', x='Coefficient', 
                hue='Indicator group', palette='tab10', 
                estimator=np.median, errorbar=lambda v: np.quantile(v,[0.1,0.9]), 
                dodge=False, capsize=0.5, width=0.95)
    
    # figure polish
    ax.set_xlim(-0.02, max(np.abs(ax.get_xlim())))
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
    
    ax.axvline(0,c='k', lw=3)
    ax.xaxis.grid(True)
    #ax.set_title('FSI indicator feature importance (predicting TMK year%i)'%(k+L-1), loc='left', fontsize=24)
    seaborn.move_legend(ax, loc='upper left')
    
    # sigh
    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)
    
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    
    fig.show()
    
    fig.savefig(f'FSI_predicting_TMK_no_ongoing_k{k}_L{L}_{tstamp}.png', bbox_inches='tight')
    fig.savefig(f'FSI_predicting_TMK_no_ongoing_k{k}_L{L}_{tstamp}.pdf', bbox_inches='tight')
    
