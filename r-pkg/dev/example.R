# requirement
library(reticulate)
install.packages("remotes")

# clean
remove.packages("ktest")
virtualenv_remove("ktest")

# python version?
py_discover_config()

# install ktest R package
remotes::install_github("AnthoOzier/ktest", ref = "rktest_dev", subdir = "r-pkg")

# configure python
venv <- "ktest"
virtualenv_create(venv)
use_virtualenv(virtualenv = venv, required = TRUE)

# python version?
py_discover_config()

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
