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
#' See the specific **"Install ktest" vignette** for more details: 
#' `vignette("install_ktest", package = "ktest")`.
#' 
#' **Important:** Python is a requirement as an intern machinery for the 
#' package to work but you will not need to create nor manipulate Python 
#' codes to use the RKeOps package.
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
#' reticulate::py_config()
#' # install pykest
#' install_ktest(method = "virtualenv", envname = "ktest")
#' }
install_ktest <- function(
        envname = NULL,
        method = c("auto", "virtualenv", "conda"),
        conda = "auto",
        python_version = NULL,
        pip = FALSE,
        ...,
        pip_ignore_installed = ignore_installed,
        ignore_installed = FALSE) {
    # check Python
    have_python <- check_python()
    # install pyktest
    reticulate::py_install(
        "ktest @ git+https://github.com/AnthoOzier/ktest@main#subdirectory=python", 
        envname = envname, 
        method = method,
        conda = conda,
        python_version = python_version,
        pip = pip,
        ...,
        pip_ignore_installed = pip_ignore_installed,
        ignore_installed = ignore_installed
    )
}
