#' Check if Python is available on the system and if its version is sufficient.
#' 
#' @keywords internal
#' 
#' @param version character string, Python required version (e.g. `"3.5"`, 
#' `3.10.2`).
#' @param warn boolean, if TRUE (default), warn user about check result.
#'
#' @return boolean value indicating if the required Python version is available 
#' on the system.
#' 
#' @importFrom reticulate py_available py_version
#' @importFrom stringr str_c
#' @importFrom utils compareVersion
check_python <- function(version = "3.5", warn = TRUE) {
    # check input
    checkmate::assert_string(
        version, pattern = "[0-9]+(\\.[0-9]+)*(\\.[0-9])?"
    )
    # Python availability?
    have_python <- reticulate::py_available()
    if(!have_python) {
        msg <- stringr::str_c(
            "Python is required but not available on your system\n",
            "Please install a Python."
        )
        if(warn) warning(msg)
        return(FALSE)
    }
    # Python version?
    if(compareVersion(as.character(reticulate::py_version()), version) < 0) {
        msg <- stringr::str_c(
            "Python version older than required ", version, "\n",
            "Please install a more recent version of Python."
        )
        if(warn) warning(msg)
        return(FALSE)
    }
    # ok
    return(TRUE)
}

#' Check if `ktest` package is ready to run.
#' 
#' @description
#' In practice, check if `ktest` Python package is installed (which is done 
#' by the function [ktest::install_ktest()]).
#' 
#' @param warn boolean, if TRUE (default), warn user about check result.
#'
#' @return boolean value indicating if the `ktest` package is ready.
#' 
#' @importFrom stringr str_c
#' 
#' @seealso [ktest::install_ktest()]
#' 
#' @examples
#' check_ktest()
#' @export
check_ktest <- function(warn = TRUE) {
    # init
    have_pyktest <- FALSE
    import_pyktest <- FALSE
    # check pyktest loading
    have_pyktest <- reticulate::py_module_available("ktest")
    if(have_pyktest) {
        import_pyktest <- tryCatch({
            tmp_pyktest <- reticulate::import("ktest")
            tmp_pyktest$READY
        }, error = function(e) return(FALSE))
    }
    if(!have_pyktest || !import_pyktest) {
        msg <- stringr::str_c(
            "'ktest' is not ready. Run `install_ktest()`."
        )
        if(warn) warning(msg)
        return(FALSE)
    } else {
        return(TRUE)
    }
}