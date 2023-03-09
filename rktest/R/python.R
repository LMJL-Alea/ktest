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

#' Check if `pyktest` Python package is available
#' 
#' @param warn boolean, if TRUE (default), warn user about check result.
#'
#' @return boolean value indicating if the `pyktest` Python package is available 
#' on the system.
#' 
#' @importFrom stringr str_c
check_pyktest <- function(warn = TRUE) {
    # init
    have_pyktest <- FALSE
    import_pyktest <- FALSE
    # check pyktest loading
    have_pyktest <- reticulate::py_module_available("pyktest")
    if(have_pyktest) {
        import_pyktest <- tryCatch({
            tmp_pyktest <- reticulate::import("pyktest")
            tmp_pyktest$READY
        }, error = function(e) return(FALSE))
    }
    if(!have_pyktest || !import_pyktest) {
        msg <- stringr::str_c(
            "'pyktest' not available. Run `install_pyktest()`."
        )
        if(warn) warning(msg)
        return(FALSE)
    } else {
        return(TRUE)
    }
}