# requirement
install.packages("remotes")
library(reticulate)

# clean (to be run only if necessary and not the first time)
remove.packages("ktest")
virtualenv_remove("ktest")

# python version?
py_discover_config()

# install ktest R package
remotes::install_github("AnthoOzier/ktest", ref = "r-ktest", subdir = "r-pkg")

# load ktest R package
library(ktest)

# configure python
venv <- "ktest"
virtualenv_create(venv)
use_virtualenv(virtualenv = venv, required = TRUE)

# python version?
py_config()

# check ktest? (should be FALSE)
check_ktest()

# install python ktest
install_ktest(method = "virtualenv", envname = venv)

# check ktest? (should be TRUE)
check_ktest()

# data
sc_df <- read.table(
    "data2.csv", row.names = 1, sep = ",", header = TRUE)
meta_sc_df <- read.table(
    "metadata2.csv", row.names = 1, sep = ",", header = TRUE)

# example
kt <- ktest(
    sc_df, meta_sc_df,
    condition='condition', samples=c('0H','48HREV'), verbose=1
)

# univariate
kt$univariate_test(
    n_jobs=4L, save_path=file.path("results"),
    name='all_variables',
    kernel=list('function'='gauss','bandwidth'='median'),
    verbose = 1L
)

# get results
kt$print_univariate_test_results(ntop=3L)

# get pval
kt$get_pvals_univariate(verbose=2L)
kt$get_pvals_univariate(t=4L, verbose=2L)

# get DE genes
kt$get_DE_genes(verbose=1L)
