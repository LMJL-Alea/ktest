#' Install Python dependency
#'
#' @description The `ktest` R package relies on the `ktest` Python package 
#' under the hood, thanks to the 
#' [`reticulate`](https://rstudio.github.io/reticulate/) package. 
#' The `ktest` Python package should be installed for R `ktest` to run.
#' This function install Python `ktest` using the `reticulate` framework.
#' 
#' @details Python should be available on your system to install and use 
#' Python `ktest`. You can run [reticulate::py_discover_config()] to check 
#' which version of Python will be used by `reticulate`.
#' 
#' To avoid messing with your system or user Python system environment, we 
#' highly recommend using a Python virtual environment or a Conda environment 
#' to install Python `ktest`. 
#' See [using Python with `reticulate`](https://rstudio.github.io/reticulate/articles/versions.html)
#' for more details.
#'
#' @inheritParams reticulate::py_install
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # create Python environement
#' reticulate::virtualenv_create("ktest")
#' # activate Python environment
#' reticulate::use_virtualenv(virtualenv = "ktest", required = TRUE)
#' # check version of Python
#' reticulate::py_discover_config()
#' # install pykest
#' install_pyktest(method = "virtualenv", envname = "ktest")
#' }
install_pyktest <- function(method = "auto", conda = "auto", ...) {
    # check Python
    have_python <- check_python()
    # install pyktest
    reticulate::py_install(
        "ktest @ git+https://github.com/AnthoOzier/ktest@rktest_dev#subdirectory=python", 
        method = method, conda = conda, ...
    )
}