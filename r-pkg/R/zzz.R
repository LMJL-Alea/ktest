#' Global reference to main ktest Python module
#' @keywords internal
#' @description
#' The `pyktest` module object is an internal reference to the ktest
#' Python package that is used under the hood by `ktest` R package. 
#'
#' @return the `pyktest` Python module
#' @usage NULL
#' @format An object of class `python.builtin.module`
pyktest <- NULL

.onLoad <- function(libname, pkgname) {
    # prepare local Python environment
    reticulate::py_require(
        packages = "ktest @ git+https://github.com/LMJL-Alea/ktest@main#subdirectory=python",
        python_version = "3.12"
    )
    # load Python ktest
    pyktest <<- reticulate::import("ktest", delay_load = TRUE)
}
