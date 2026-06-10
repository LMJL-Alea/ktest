import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch as to

# from ktest.data import Data
# from ktest.kernel_statistics import Statistics
from ktest.tester import Ktest


# data shape
data_shape = [1000, 1000]

# generate random data with different mean in different the 2 groups
rng = np.random.default_rng(42)
loc = 0
loc = np.hstack([
    np.tile(
        np.array([-2, 2])[np.arange(data_shape[0]) % 2][:, np.newaxis],
        (1, 10)
    ) + rng.normal(loc=0, scale=1, size=[data_shape[0], 10]),
    np.zeros([data_shape[0], data_shape[1] - 10], dtype=np.float64)
])
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
plt.savefig("fig/data.png")

# Perform PCA
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

# plot PCA
data2plot = pd.DataFrame(transformed_data, columns=['PC1', 'PC2'])
data2plot["meta"] = meta
plt.figure()
sns.scatterplot(data=data2plot, x='PC1', y='PC2', hue="meta")
plt.savefig("fig/pca.png")

# Kernel test
kt = Ktest(
    data=data, metadata=meta, nystrom=False, dtype=to.float64
)
kt.test(verbose=0)

print(f"pvalues =\n{kt.kfda_pval_asymp.to_numpy().T[:10]}")

# plot density
plt.figure()
kt.plot_density(t=10)
plt.savefig("fig/density.png")

# projection
proj_kfda, proj_kpca = kt.project(
    t=100, center=True, verbose=1, new_obs=None
)

# plot projection
data2plot = pd.concat(proj_kfda.values()).sort_index().rename(
    lambda x: f"t{x}", axis="columns"
)
data2plot["meta"] = meta

for t_val in [5, 10, 20, 50, 75, 100]:
    plt.figure()
    sns.kdeplot(data=data2plot, x=f"t{t_val}", hue="meta")
    plt.savefig(f"fig/proj_t{t_val}.png")

# cross-validation
accuracy, true_pos, true_neg, residuals = kt.cv(
    t=100, pred_threshold=1/2, n_fold=5, n_repeat=1,
    random_state=None, verbose=1
)

# plot prediction performance
res_df = pd.DataFrame({
    "accuracy": accuracy[0],
    "true_pos": true_pos[0],
    "true_neg": true_neg[0],
    "t_val": np.arange(100)+1
})

data2plot = pd.melt(res_df, id_vars=["t_val"])

plt.figure()
sns.lineplot(
    x='t_val',
    y='value',
    hue='variable',
    data=data2plot
)
plt.savefig(f"fig/cv_res.png")

# plot residuals
res_df = pd.DataFrame({
    "accuracy": accuracy[0],
    "true_pos": true_pos[0],
    "true_neg": true_neg[0],
    "t_val": np.arange(100)+1
})

data2plot = pd.DataFrame({
    "residuals": residuals[0],
    "t_val": np.arange(100)+1
})

plt.figure()
sns.lineplot(
    x='t_val',
    y='residuals',
    data=data2plot
)
plt.savefig(f"fig/cv_residuals.png")



