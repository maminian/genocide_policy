
import load_ewp_plus_flags as _duh
df = _duh.df

from sklearn import decomposition
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('bmh')

# Of particular interest are predicting any or all of the last 
# three labels:
#
# coercive_control
# existential_threat
# id_target_gps        (identified target groups)
#

X = df.iloc[:,3:-3]
X.drop('SFTGcode', axis=1, inplace=True)
y = df.iloc[:,-3:]

#pca = decomposition.PCA()
pca = decomposition.SparsePCA(n_components=2)
#pca = decomposition.KernelPCA(n_components=2, kernel='rbf')


Xd = pca.fit_transform(X)

fig,ax = plt.subplots(1,3, figsize=(12,4), sharex=True, sharey=True, constrained_layout=True)

for i in range(3):
    ax[i].scatter(
        Xd[:,0], Xd[:,1], c=y.iloc[:,i], 
        label=df.columns[-3+i], cmap=plt.cm.cividis, edgecolors='k'
    )
    
    ax[i].set(xlabel='PCA1')
    ax[i].legend()

ax[0].set(ylabel='PCA2')

fig.show()

fig2,ax2 = plt.subplots(constrained_layout=True)
ypos = np.arange(X.shape[1])
ypos = ypos[::-1]

ax2.barh(ypos, pca.components_[0], label='PCA1 weights')
ax2.barh(ypos, pca.components_[1], label='PCA2 weights')

ax2.legend()
ax2.set(xlabel='Principal component weight', yticks=[])

fig2.show()


