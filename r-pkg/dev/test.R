# requirement
library(reticulate)

# python setup?
py_discover_config()

# load ktest R package
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "r-pkg")
devtools::load_all(pkg_dir)

venv <- "ktest-dev"
virtualenv_create(venv)
use_virtualenv(virtualenv = venv, required = TRUE)
py_config()

# check
check_ktest()

# install python ktest
install_ktest(method = "virtualenv", envname = venv)

# check
check_ktest()

# data
sc_df <- read.table("data/data2.csv", row.names = 1, sep = ",", header = TRUE)
meta_sc_df <- read.table("data/metadata2.csv", row.names = 1, sep = ",", header = TRUE)

# test
kt <- ktest(
    sc_df, meta_sc_df,
    condition='condition', samples=c('0H','48HREV'), verbose=1
)

# multivariate
kt$multivariate_test(verbose=1)

# univariate
kt$univariate_test(
    n_jobs=4L, save_path="./",
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
