# requirements
library(conflicted)     # manage namespace conflict between packages
library(reticulate)     # manage Python dependencies

library(dplyr)          # manage data.frame
library(readr)          # load data.frame
library(stringr)        # process character string
library(tibble)         # manipulate data.frame


#---- Install and configuration (to be done once!) ----------------------------#
# install ktest
install.package("remotes")
remotes::install_github("LMJL-Alea/ktest", ref = "main", subdir = "r-pkg")
# load ktest R package
library(ktest)
# create dedicated Python virtual environment
reticulate::virtualenv_create("ktest")
# activate the python environment
reticulate::use_virtualenv(virtualenv = "ktest", required = TRUE)
# verify python version
reticulate::py_config()
# install ktest package python requirements
install_ktest(method = "virtualenv", envname = "ktest")
# check ktest configuration
check_ktest()

#---- loading the package for regular use -------------------------------------#
library(ktest)
reticulate::use_virtualenv(virtualenv = "ktest", required = TRUE)

check_ktest()


#---- Data import -------------------------------------------------------------#
# download dataset
data_tab <- readr::read_csv(
    "https://raw.githubusercontent.com/LMJL-Alea/ktest/main/tutorials/v5_data/RTqPCR_reversion_logcentered.csv",
    show_col_types = FALSE
)

# extract sample condition
metadata_tab <- data_tab %>% pull(1) %>% 
    str_split(pattern = "\\.", simplify = TRUE) %>%
    as_tibble(.name_repair = "universal") %>% select(2) %>% 
    rename_with(~c("condition"))

# drop sample condition from gene expression table
data_tab <- data_tab %>% select(!1)

# data dimension (cells x genes)
dim(data_tab)

# detail sample conditions
table(metadata_tab)

#---- kernel test -------------------------------------------------------------#

# initialize ktest object
kt_1 = ktest_init(
    data = data_tab, metadata = metadata_tab, 
    sample_names = c('0H','48HREV')
)

# run test
test(
    kt = kt_1, 
    stat = 'kfda', 
    permutation = TRUE, 
    n_permutations = 500, 
    verbose = 1
)

# get statistic values
get_statistics(kt_1, stat = 'kfda', contrib = FALSE, t_max = 50)

# get p-values
get_pvalues(kt_1, stat = 'kfda', permutation = FALSE, t_max = 50)

# get kernel embedding projections
proj <- get_proj(kt_1, contrib = FALSE, t_max = 50)
