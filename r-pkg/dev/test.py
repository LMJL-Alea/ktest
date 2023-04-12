import pandas as pd
from ktest.tester import Ktest

# data
data = pd.read_csv('data.csv',index_col=0)
metadata = pd.read_csv('metadata.csv',index_col=0)

# test
kt = Ktest(data, metadata, samples=['0H','48HREV'], condition='condition', nystrom=True)

# univariate
kt.univariate_test(
    n_jobs=4,lots=20,verbose=1,name='all_variables',
    save_path='./'
)

# get results
kt.print_univariate_test_results(ntop=3)

# get pval
kt.get_pvals_univariate(verbose=2)
kt.get_pvals_univariate(t=4, verbose=2)

# get DE genes
kt.get_DE_genes(verbose=1)
