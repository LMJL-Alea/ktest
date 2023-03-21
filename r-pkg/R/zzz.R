# global reference to pyktest (will be initialized in .onLoad)
pyktest <- NULL

.onLoad <- function(libname, pkgname) {
    if(check_pyktest(warn = FALSE)) {
        # use superassignment to update global reference to pyktest
        pyktest <<- reticulate::import("ktest", delay_load = TRUE)
    }
}