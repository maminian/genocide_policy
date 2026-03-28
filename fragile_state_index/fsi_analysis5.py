
import pandas
import numpy as np

import load_fsi
import vis_tools_fsi
import load_tmk
import datasets

#

from sklearn import linear_model # for Lasso, etc.
from sklearn import ensemble     # for Random Forest, etc.
from sklearn import metrics      # for, e.g., sklearn.metrics.roc_auc_score
from sklearn import model_selection

#
#

import datetime
tstamp = datetime.datetime.now().strftime('%Y-%b-%d-%H:%M')

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
countries_flat = np.concatenate(meta['countries'])
years_flat = np.concatenate(meta['years'])


# final labeled data set (!!)

###############

'''
Proper train/test scheme now proceeds as follows:
    
    select ntmk samples from each class.
    subdivide into training testing sets; ns from each class in training.
    Train; then evaluate and store probabilities for later evaluation of results.

    Last issue: this naive approach breaks causality (future events may be 
    in the training set to predict past events)
'''

# Number of positive/negative predictive samples per bootstrap
ns = 15

not_tmk_idx = np.where(y==0)[0]
yes_tmk_idx = np.where(y>0)[0]
ntmk = len(yes_tmk_idx)
print("k: %i, L: %i, ntmk: %i"%(k,L,ntmk) )

np.random.seed(10072023)
# number of bootstrap samples
nboots = 10000
models = []
models_coef_ = np.zeros( (nboots, 12*k) )

subsets = np.zeros((nboots, 2*ntmk), dtype=int)
trains = np.zeros((nboots, ns))
tests = np.zeros((nboots, ns))
preds_lr = np.zeros((nboots, ns))
preds_rf = np.zeros((nboots, ns))
aucrocs_lr = np.zeros(nboots)

train_idxs = np.zeros((nboots, ns), dtype=int)
test_idxs = np.zeros((nboots, ns), dtype=int)

for i in range(nboots):
    if i%(nboots//100)==0:
        print(f'{i} of {nboots}')
    model = linear_model.LogisticRegression(max_iter=4000, solver='saga', l1_ratio=0.05)
    model2 = ensemble.RandomForestClassifier()
    
    #not_tmk_idx_choice = np.random.choice(not_tmk_idx, ntmk, replace=False)
    #subset = np.concatenate( [yes_tmk_idx, not_tmk_idx_choice] )
    
    #subsets[i] = subset

    
    # force a balanced stratified selection by first manually subsetting data.
    not_tmk_idx_choice = np.random.choice(not_tmk_idx, ntmk, replace=False)
    _X2 = np.vstack([X[not_tmk_idx_choice], X[yes_tmk_idx]])
    _y2 = np.hstack([y[not_tmk_idx_choice], y[yes_tmk_idx]])
    _all_idx = np.hstack([not_tmk_idx_choice, yes_tmk_idx])
    
    # now do stratified subsampling.
    _sel = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=ns, train_size=ns)

    _split = list( _sel.split(_X2, _y2) )

    #import pdb
    #pdb.set_trace()
    
    train_idx = _all_idx[_split[0][0]]
    test_idx = _all_idx[_split[0][1]]
    
    
    #train_idx = np.concatenate([np.random.choice(yes_tmk_idx, ns, replace=False), np.random.choice(not_tmk_idx_choice, ns, replace=False)])
    #test_idx = np.setdiff1d(subset, train_idx)
    
    train_idxs[i] = train_idx
    test_idxs[i] = test_idx
    
    model.fit(X[train_idx], y[train_idx])
    ypred = model.predict_proba(X[test_idx])[:,1]
    
    model2.fit(X[train_idx], y[train_idx])
    
    trains[i] = y[train_idx]
    tests[i] = y[test_idx]
    preds_lr[i] = ypred # note: float (probabilities/scores)
    preds_rf[i] = model2.predict(X[test_idx]) # note: integer
    # can do more sophisticated things later...

    aucrocs_lr[i] = metrics.roc_auc_score(tests[i], preds_lr[i])
    
    models.append(model)
    models_coef_[i] = model.coef_
    
    #print(i, '%.3f'%aucrocs[i])
#

# build long dataframe solely for the purposes of visualization.
df_results = pandas.DataFrame(data=models_coef_,columns=features).melt(var_name='Indicator', value_name='Coefficient')
df_results['Indicator_group'] = [{'X':'S'}.get(v[0],v[0]) for v in df_results['Indicator']]


############
# build dataframe to export results.

# table 1...
# country, year, tmk label, predicted probability, bootstrap number
columns = ['country', 'year', 'true_label', 'pred_prob_lr', 'pred_lr', 'pred_rf',  'bootstrap_number']
_cc = countries_flat[ test_idxs ].flatten()
_yy = years_flat[ test_idxs ].flatten()
_cy = [(_cc[i], _yy[i]) for i in range(len(_cc))]
_tt = np.array(tests.flatten(), dtype=int)
_pp_prob_lr = preds_lr.flatten()
_pp_lr = np.array(_pp_prob_lr>0.5, dtype=int)
_pp_rf = np.array(preds_rf.flatten(), dtype=int)
_bb = np.repeat(np.arange(nboots), ns)

df_crossval = pandas.DataFrame({header: dat for (header,dat) in zip(columns, [_cc, _yy, _tt, _pp_prob_lr, _pp_lr, _pp_rf, _bb])})


df_crossval.to_csv(f"bootstrap_pred_results_k{k}_L{L}_{tstamp}.csv", index=False)
    

# table 2...
# bootstrap number, (country1, year1), (country2, year2), ..., (countryM, yearM) in training data.



############

#
if __name__=="__main__":

    from matplotlib import pyplot as plt

    plt.rcParams.update({'font.size': 16})
    plt.style.use('seaborn-v0_8-whitegrid')


    all_curves = [metrics.roc_curve(tests[i],preds_lr[i], drop_intermediate=False) for i in range(nboots)]
    all_curves = np.array(all_curves)
    # plot performance measured by AUC ROC
    fig,ax = plt.subplots(1,2, figsize=(12,6), constrained_layout=True)
    
    # plot an arbitrary subset of ROC curves
    #for i in range(0, nboots, int(nboots/10000)):
    #    ax[0].plot(all_curves[i][0], all_curves[i][1], c='#666', alpha=0.1, lw=4)
    ax[0].plot(all_curves[0,0,:], all_curves[:,1,:].T, c='#666', alpha=0.1, lw=4)
    ax[1].hist(aucrocs_lr, bins=np.linspace(0,1,41), edgecolor='k', linewidth=0.5)
    
    ax[0].set(xlim=(0,1), ylim=(0,1), xlabel="FPR", ylabel="TPR")
    ax[1].set(xlim=(0,1), xlabel="AUCROC", ylabel="Count")
    
    fig.savefig(f"bootstrap_traintest_lr_auc_pred_k{k}_L{L}_{tstamp}.png", bbox_inches='tight')
    fig.savefig(f"bootstrap_traintest_lr_auc_pred_k{k}_L{L}_{tstamp}.pdf", bbox_inches='tight')
    
