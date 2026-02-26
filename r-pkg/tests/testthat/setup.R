# requirements
skip_if_not_installed("checkmate")
skip_if_not_installed("dplyr")
skip_if_not_installed("fs")
skip_if_not_installed("readr")
skip_if_not_installed("reticulate")
skip_if_not_installed("tibble")
skip_if_not_installed("withr")

library(checkmate)
library(dplyr)
library(fs)
library(readr)
library(reticulate)
library(tibble)
library(withr)

# helper function to skip tests if Python is not available on the system
skip_if_no_python <- function() {
    have_python <- reticulate::py_available()
    if(!have_python) skip("Python not available on system")
}

# helper function to skip tests if `pyktest` is not available
skip_if_no_pyktest <- function() {
    have_pyktest <- reticulate::py_module_available("ktest")
    if(!have_pyktest) skip("'ktest' Python package not available for testing")
}
