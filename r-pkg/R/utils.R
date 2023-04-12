#' Helper function to generate random python environment name
#' 
#' @keywords internal
#'
#' @param prefix character string, prefix to add to the environment name.
#' @param length integer, number of random character in the environment name.
#'
#' @return character string, name of the python environment.
#' 
#' @importFrom checkmate assert_string assert_count
#' @importFrom stringi stri_rand_strings
#'
#' @examples
#' random_envname(prefix = "test", length = 10)
random_envname <- function(prefix = "", length = 10) {
    checkmate::assert_string(prefix)
    checkmate::assert_count(length)
    envname <- stringr::str_c(
        prefix,
        stringi::stri_rand_strings(1, length)
    )
    return(envname)
}