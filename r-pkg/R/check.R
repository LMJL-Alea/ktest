#' Check Python availability
#' 
#' @description
#' Check if Python is available on the system and if its version is sufficient.
#' 
#' @details
#' This function is just a wrapper around [reticulate::py_available()] and 
#' [reticulate::py_version()] functions.
#' 
#' @param version character string, Python required version (e.g. `"3.5"`, 
#' `"3.10.2"`).
#' @param verbose boolean, if TRUE (default), inform user about check result.
#'
#' @return boolean value indicating if the required Python version is available 
#' on the system.
#' 
#' @importFrom checkmate assert_string assert_flag
#' @importFrom reticulate py_available py_version
#' @importFrom stringr str_c
#' @importFrom utils compareVersion
#' 
#' @export
#' @examples
#' \dontrun{
#' check_python()
#' }
check_python <- function(version = "3.5", verbose = TRUE) {
    # check input
    assert_string(version, pattern = "[0-9]+(\\.[0-9]+)*")
    assert_flag(verbose)
    # Python availability?
    have_python <- reticulate::py_available()
    if(!have_python) {
        msg <- str_c(
            "\nATTENTION: ",
            "Python is required but not available on your system.\n\n",
            "Please verify Python availability on your system ", 
            "with 'reticulate::py_discover_config()' and ", 
            "'reticulate::py_available()' functions from the 'reticulate' ",
            "package.\n\n",
            "You can also check the 'reticulate' package documentation at ",
            "https://rstudio.github.io/reticulate/ for more details.\n",
            "Note: we recommend that you use ",
            "a dedicated Python environment, see 'reticulate' documentation ",
            "or 'ktest' package vignette or README."
        )
        if(verbose) warning(msg)
        return(FALSE)
    }
    # Python version?
    if(compareVersion(as.character(reticulate::py_version()), version) < 0) {
        msg <- str_c(
            "Python version older than required ", version, "\n",
            "Please install a more recent version of Python.",
            "You can also check the 'reticulate' package documentation at ",
            "https://rstudio.github.io/reticulate/ for more details.\n"
        )
        if(verbose) warning(msg)
        return(FALSE)
    }
    # ok
    msg <- str_c(
        "Python ", as.character(reticulate::py_version()), " is available."
    )
    if(verbose) message(msg)
    return(TRUE)
}

#' Check if `ktest` package is ready to run.
#' 
#' @description
#' In practice, check if `ktest` Python package is installed (which should be
#' done by the [ktest::install_ktest()] function).
#' 
#' @param verbose boolean, if TRUE (default), inform user about check result.
#'
#' @return boolean value indicating if the `ktest` package is ready.
#' 
#' @importFrom stringr str_c
#' 
#' @seealso [ktest::install_ktest()]
#' 
#' @examples
#' \dontrun{
#' check_ktest()
#' }
#' @export
check_ktest <- function(verbose = TRUE) {
    # init
    have_python <- FALSE
    have_pyktest <- FALSE
    import_pyktest <- FALSE
    # check pyton
    have_python <- check_python(version = "3.5", verbose = FALSE)
    if(!have_python) return(FALSE)
    # check pyktest loading
    have_pyktest <- reticulate::py_module_available("ktest")
    if(have_pyktest) {
        import_pyktest <- tryCatch({
            tmp_pyktest <- reticulate::import("ktest")
            tmp_pyktest$READY
        }, error = function(e) return(FALSE))
    }
    if(!all(have_python, have_pyktest, import_pyktest)) {
        msg <- stringr::str_c(
            "\nATTENTION: 'ktest' is not ready.\n\n",
            "You should:\n",
            "1. verify that Python is available on your system, ", 
            "see 'reticulate::py_discover_config()' and ", 
            "'reticulate::py_available()' functions from the 'reticulate' ",
            "package,\n",
            "2. restart your R session,\n",
            "3. run the function 'install_ktest()' after ", 
            "loading 'ktest'.\n\n",
            "Note: we recommend that you use ",
            "a dedicated Python environment, see 'reticulate' documentation ",
            "or 'ktest' package vignette or README.\n\n",
            "You can also check the 'reticulate' package documentation at ",
            "https://rstudio.github.io/reticulate/ for more details.\n")
        if(verbose) warning(msg)
        return(FALSE)
    } else {
        msg <- stringr::str_c("'ktest' is ready.")
        if(verbose) message(msg)
        return(TRUE)
    }
}