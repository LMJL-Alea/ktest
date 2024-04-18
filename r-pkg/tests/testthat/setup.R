# requirements
skip_if_not_installed("checkmate")
skip_if_not_installed("dplyr")
skip_if_not_installed("fs")
skip_if_not_installed("readr")
skip_if_not_installed("reticulate")
skip_if_not_installed("tibble")

library(checkmate)
library(dplyr)
library(fs)
library(readr)
library(reticulate)
library(tibble)

# helper function to skip tests if Python is not available on the system
skip_if_no_python <- function() {
    have_python <- reticulate::py_available()
    if(!have_python) skip("Python not available on system")
}

# helper function to skip tests if `pyktest` is not available
skip_if_no_pyktest <- function() {
    have_pyktest <- reticulate::py_module_available("pyktest")
    if(!have_pyktest) skip("'pyktest' not available for testing")
}

# helper function to skip tests if not interactive mode
skip_if_not_interactive <- function() {
    if(!interactive()) skip("Test only run in interactive mode")
}

# function to load test data
load_test_data <- function() {
    # data directory
    data_dir <- file.path(fs::path_package("ktest"), "extdata")
    # expression data table
    data_tab <- readr::read_csv(file.path(data_dir, "data.csv")) %>%
        dplyr::select(!1)
    # metadata table
    metadata_tab <- readr::read_csv(file.path(data_dir, "metadata.csv")) %>%
        dplyr::select(condition)
    # output
    return(tibble::lst(data_tab, metadata_tab))
}
