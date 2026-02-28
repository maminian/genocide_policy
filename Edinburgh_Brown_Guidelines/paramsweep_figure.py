import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker

import numpy as np

sns.set_theme(style='whitegrid')

LR_COLOR = '#f9fff9' # pale pink/purple
RF_COLOR = '#fff9ff' # pale green

WHICH = 'f1' # acc or f1

#################

df_paramsweep = pd.read_csv('ebg_ewp_expt_18feb2026.csv')
df_no_shuffle = df_paramsweep[df_paramsweep['shuffled']=='N']



for _suffix,_df_ref in zip(['orig','vs_shuffle'], [df_no_shuffle, df_paramsweep]):

    # "standard" box plot...
    # box is the interquartile range (percentiles [25,75])
    # whiskers halt at the furthest data point at most 
    # 1.5IQR lengths above/below the 25th and 75th percentiles.
    # the remainder beyond this range are plotted individually.
    # 
    g = sns.catplot(data=_df_ref, 
                    kind='box', col='code', row='classifier',
                    x='test_size', y=WHICH, hue='shuffled', 
                    fill=True,
                    gap=.2,
                    margin_titles=True,
                    flierprops={"marker": ".", "c":[0,0,0,0.4]},
                    #gridspec_kw={'hspace':0.1}
                    #facet_kws={'gridspec_kws':{'hspace':0.}} # ew lol
                   )

    fig = g.figure
    ax = g.axes

    # polish...
    for j in range(np.shape(ax)[1]):
        # strip the "code = ..."
        ax[0,j].set_title(ax[0,j].get_title().split('=')[-1].strip())
        ax[1,j].set(xlabel='test size')
        # write the sizes as percentages.
        ax[1,j].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p: f'{10+int(x*10)}%'))
        
    for i in range(np.shape(ax)[0]):
        _t = ax[i,-1].texts[0]
        # strip the "classifier = ..." and add color-coding.
        _t.set(
            text=_t.get_text().split('=')[-1].strip(),
            fontweight='bold',
            bbox={'fc':[LR_COLOR,RF_COLOR][i], 'ec':'#333'}
        )
        
        # "acc" -> "accuracy"; "f1" -> "F1 score"
        ax[i,0].set(ylabel={'acc':'accuracy', 'f1':'F1 score'}[WHICH])

    # background color to signal classifier...?
    for j in range(np.shape(ax)[1]):
        ax[0,j].set_facecolor(LR_COLOR)
        ax[1,j].set_facecolor(RF_COLOR)
        if WHICH=='acc':
            for i in range(np.shape(ax)[0]):
                ax[i,j].set(ylim=(0.4,1.05))

    fig.subplots_adjust(hspace=0.2)

    fig.show()
    fig.savefig(f'paramsweep_results_{WHICH}_{_suffix}.png', bbox_inches='tight')
    fig.savefig(f'paramsweep_results_{WHICH}_{_suffix}.pdf', bbox_inches='tight')
    
