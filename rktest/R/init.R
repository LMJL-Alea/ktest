#' Install Python dependency
#'
#' @description The `rktest` package relies on `pyktest` Python package under 
#' the hood, thanks to the 
#' [`reticulate`](https://rstudio.github.io/reticulate/) package. 
#' The `pyktest` package should be install for `rktest` to run.
#' This function install `pyktest` using the `reticulate` framework.
#' 
#' @details Python should be available on your system to install `pyktest`. You 
#' can run [reticulate::py_discover_config()] to check which version of Python 
#' will be used by `reticulate`.
#' 
#' To avoid messing with your system or user Pytheon environment, we highly 
#' recommend using a Python virtual environment or a Conda environment to 
#' install `pyktest`. See [using Python with `reticulate`](https://rstudio.github.io/reticulate/articles/versions.html)
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
        "pyktest @ git+https://github.com/AnthoOzier/ktest@rktest_dev#subdirectory=pyktest", 
        method = method, conda = conda, ...
    )
}