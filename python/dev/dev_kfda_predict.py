import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch as to

from ktest.data import Data
from ktest.kernel_statistics import Statistics
from ktest.tester import Ktest


# data shape
data_shape = [1000, 100]

# generate random data with different mean in different the 2 groups
rng = np.random.default_rng(42)
loc = 0
# loc = np.hstack([
#     np.tile(
#         np.array([-2, 2])[np.arange(data_shape[0]) % 2][:, np.newaxis],
#         (1, 10)
#     ) + rng.normal(loc=0, scale=1, size=[data_shape[0], 10]),
#     np.zeros([data_shape[0], data_shape[1] - 10], dtype=np.float64)
# ])
data_array = rng.normal(loc=loc, scale=2, size=data_shape)

# create a data frame from random gaussian data
data = pd.DataFrame(
    data=data_array,
    columns=[f"col{i+1}" for i in range(data_shape[1])]
)

# create meta data frame indicating two groups
meta = pd.Series(
    data=[f"c{i+1}" for i in range(2)] * (data_shape[0] // 2)
)

# heatmap of the data matrix
plt.figure()
sns.heatmap(pd.concat(
    [data.iloc[::2], data.iloc[1::2]], ignore_index=True
))
plt.savefig("data.png")

# Perform PCA
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

# plot PCA
data2plot = pd.DataFrame(transformed_data, columns=['PC1', 'PC2'])
data2plot["meta"] = meta
plt.figure()
sns.scatterplot(data=data2plot, x='PC1', y='PC2', hue="meta")
plt.savefig("pca.png")

# Kernel test
kt = Ktest(
    data=data, metadata=meta, nystrom=False, dtype=to.float64
)
kt.test(verbose=0)

print(f"pvalues =\n{kt.kfda_pval_asymp.to_numpy().T[:10]}")

# plot density
plt.figure()
kt.plot_density(t=10)
plt.savefig("density.png")

# ktest data
data_obj = Data(
    data=data,
    metadata=meta,
    sample_names=None,
    nystrom=False,
    dtype=to.float64
)

# kernel stat object
kstat = Statistics(
    data=data_obj,
    kernel_function='gauss',
    bandwidth='median',
    median_coef=1,
    eps=None, clip_eigval=True
)

# compute stat
stat_val, _ = kstat.compute_kfda()

print(f"Stat value =\n{stat_val.to_numpy().T[:10]}")

# compute projections
proj_kfda, proj_kpca = kstat.compute_projections(
    stat=stat_val, t=50, center=False
)

# plot projection
data2plot = pd.concat(proj_kfda.values()).sort_index().rename(
    lambda x: f"t{x}", axis="columns"
)
data2plot["meta"] = meta
plt.figure()
sns.kdeplot(data=data2plot, x="t50", hue="meta")
plt.savefig("proj.png")

# prediction
pred, loss = kstat.kfda_predict(t=50, pred_threshold=0)

# compute accuracy
for group in kstat.data.data.keys():
    n_obs = kstat.data.data[group].shape[0]
    count_pred = np.count_nonzero(pred[group] == group, axis=0)
    print(f"Prediction accuracy for group '{group}': {count_pred / n_obs}")

