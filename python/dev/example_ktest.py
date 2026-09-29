import pandas as pd
from ktest.tester import Ktest

# data
url = "https://raw.githubusercontent.com/LMJL-Alea/ktest/main/tutorials/v5_data/RTqPCR_reversion_logcentered.csv"
data = pd.read_csv(url, index_col=0)

meta = pd.Series(data = pd.Series(data.index).apply(lambda x : x.split(sep='.')[1]))
meta.index = data.index

# test
kt_1 = Ktest(data=data, metadata=meta, sample_names=['48HREV','48HDIFF'], nystrom=True)
kt_1.test()
print(kt_1)

kt_1.save("ktest_RTqPCR_reversion_logcentered.pkl")



import matplotlib.pyplot as plt
plt.figure()
kt_1.plot_density(trunc=10)
plt.savefig("test_ktest_density.png")

plt.figure()
kt_1.scatter_projection()
plt.savefig("test_ktest_proj.png")
