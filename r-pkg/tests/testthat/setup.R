# helper function to run code in a dedicated python environment
# (with a random name and that will be removed afterwards)
run_py_env <- function(code, envname = NULL) {
    # new envname if not provided
    if(is.null(envname)) {
        envname <- random_envname("ktest", length = 20)
    }
    # check if environment exists
    if(!reticulate::virtualenv_exists(envname)) {
        # previous Python (if available)
        prev_python <- reticulate::py_exe()
        # create temp Python environment
        reticulate::virtualenv_create(envname)
        withr::defer({
            reticulate::use_python(prev_python)
            reticulate::virtualenv_remove(envname, confirm = FALSE)
        })
        # activate Python environment
        reticulate::use_virtualenv(virtualenv = envname, required = TRUE)
        # run code
        force(code)
    } else {
        # if environment exist (unlikely), then error
        msg <- stringr::str_c(
            "'", envname, "' random package name already exists, ",
            "which should not be the case."
        )
        stop(msg)
    }
}

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
