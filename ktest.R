# use ktest package in R (quick and dirty)

## requirements
library(reticulate)
library(readr)
library(tidyverse)


py_discover_config()

## use system python
use_python("/usr/bin/python")


## check python version
py_config()


## create virtualenv
virtualenv_create("ktest")

## install Python package (to do once)
virtualenv_install(
    envname = "ktest",
    packages = "pyktest @ git+https://github.com/AnthoOzier/ktest@rktest_dev#subdirectory=pyktest"
)

## activate python environment
use_virtualenv(virtualenv = "ktest", required = TRUE)

py_discover_config()
py_config()

## Python import
pd <- import("pandas",as = "pd")
pyktest <- reticulate::import("pyktest")

#### Univariate test vignette
sc_df <- read.table("data/data.csv", row.names = 1, sep = ",", header = TRUE)
str(sc_df)
rownames(sc_df)

meta_sc_df <- read.table("data/metadata.csv", row.names = 1, sep = ",", header = TRUE)
str(meta_sc_df)
rownames(meta_sc_df)


#### Multivariate test
Ktest <- py_run_string("from pyktest.tester import Ktest")
kt <- pyktest$tester$Ktest(
    sc_df, meta_sc_df,
    condition='condition',
    samples=c('0H','48HREV'),
    verbose=1)

kt$multivariate_test(verbose=1)
kt$print_multivariate_test_results(long=TRUE,ts=c(1,2,3))
