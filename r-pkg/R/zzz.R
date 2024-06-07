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
    pyktest <<- reticulate::import("ktest", delay_load = TRUE)
}
