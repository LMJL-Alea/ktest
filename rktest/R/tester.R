#' Generate a Ktest object and compute specified comparison
#' 
#' @param data `data.frame` or equivalent containing the data with 
#' observation in rows and variables/features in columns.
#' 	
#' @param metadata `data.frame` or equivalent containing the metadata with 
#' observation in rows and variables/features in columns.
#' 
#' @param condition : character string, column name of the metadata 
#' `data.frame` containing the labels to test
#' 
#' @param data_name character string, name of the data (default = `"data"`, 
#' examples: `"counts"`, `"normalized"`)
#' 
#' @param samples character string or vector/list of character string 
#' (default = `"all"`):
#' - character string: category of condition to compare to others as 
#'   + `"all"`: test the two categories contained in the column condition
#' (does not work for more than two yet)
#'   + `"one vs all"`
#' - vector/list of character string: the two categories to compare to 
#' each other.
#' 	
#' @param var_metadata `data.frame` or equivalent containing metainformation 
#' about the variables, index must correspond to `data` columns 
#' (default = `NULL`).
#' 
#' @param stat character string among `"kfda"`, `"mmd"` (default = `"kfda"`).
#' 	The test statistic to use. If `stat == "mmd"`, then 
#' 	`permutation` is set to `TRUE`.
#' 
#' @param permutation boolean, indicating whether to compute permutation or 
#' asymptotic p-values(default = `FALSE`). Set to `TRUE` if `stat == "mmd"`.
#' 
#' @param n_permutations integer (default = `500`), number of permutations needed 
#' to compute the permutation p-value. Ignored if `permutation` is `FALSE`. 
#' 
#' @param seed_permutation integer (default = `0`), seed for the first 
#' permutation. The other seeds are obtained by incrementation. 
#' 
#' @param nystrom boolean (default = `FALSE`). If `TRUE`, the `test_params` 
#' input argument is set to `TRUE` with default parameters.
#' 
#' @param test_params object specifying the test to execute, 
#' output of the function `get_test_params()`.
#' 
#' @param center_by character string (default = `NULL`),
#' 	column of the metadata containing a categorical variable to regress 
#' 
#' @param marked_obs_to_ignore character string (default = `NULL`), 
#' column of the metadata containing boolean values that are 
#' `TRUE` if the observation should be ignored from the analysis 
#' and `FALSE` otherwise.
#' 
#' @param kernel : object specifying the kernel function to use
#' output of the function `init_kernel_params()`.
#' 	
#' @param verbose integer (`1` or `0`) or logical (`TRUE` or `FALSE`), 
#' enable/disable verbosity.
#'
#' @return a `Ktest` object.
#' @export
#'
#' @examples
#' # WRITE ME
ktest <- function(
        data,
        metadata,
        condition,
        data_name='data',
        samples='all',
        var_metadata=NULL,
        stat='kfda',
        nystrom=FALSE,
        permutation=FALSE,
        n_permutations=500,
        seed_permutation=0,
        test_params=NULL,
        center_by=NULL,
        marked_obs_to_ignore=NULL,
        kernel=NULL,
        verbose=0
) {
    # check input
    assert_data_frame(data)
    assert_data_frame(metadata)
    assert_string(condition)
    assert_string(data_name)
    assert_character(samples, min.len = 1)
    assert_data_frame(var_metadata, null.ok = TRUE)
    assert_choice(stat, c("kfda", "mmd"))
    assert_logical(nystrom, len = 1)
    assert_logical(permutation, len = 1)
    assert_count(n_permutations, positive = TRUE)
    assert_count(seed_permutation)
    # TODO: check test_params
    assert_string(center_by, null.ok = TRUE)
    assert_string(marked_obs_to_ignore, null.ok = TRUE)
    # TODO: check kernel
    qassert(verbose, c("B1", "X1[0,1]"))
    
    # Python call
    return(pyktest$tester$Ktest(
        data,
        metadata,
        condition,
        data_name,
        samples,
        var_metadata,
        stat,
        nystrom,
        permutation,
        n_permutations,
        seed_permutation,
        test_params,
        center_by,
        marked_obs_to_ignore,
        kernel,
        verbose
    ))
}

#' Compute multivariate testing
#'
#' @param kt output of [`rktest::ktest()`] function.
#' @inheritParams ktest
#' @return No return.
#' @export
#'
#' @examples
#' # WRITE ME
multivariate_test <- function(
        kt, 
        stat=NULL,
        permutation=NULL,
        n_permutations=500,
        seed_permutation=0,                   
        n_jobs_permutation=1,
        keep_permutation_statistics=FALSE,
        verbose=0
) {
    # check input
    assert_choice(stat, c("kfda", "mmd"), null.ok = TRUE)
    assert_logical(permutation, len = 1, null.ok = TRUE)
    assert_count(n_permutations, positive = TRUE)
    assert_count(seed_permutation)
    assert_count(n_jobs_permutation)
    assert_logical(keep_permutation_statistics, len = 1)
    qassert(verbose, c("B1", "X1[0,1]"))
    
    if(is.null(stat)) stat <- kt$stat
    if(is.null(permutation)) permutation <- kt$permutation
    if(is.null(n_permutations)) n_permutations <- kt$n_permutations
    if(is.null(seed_permutation)) seed_permutation <- kt$seed_permutation
    
    # Python call
    return(kt$multivariate_test(
        stat,
        permutation,
        n_permutations,
        seed_permutation,                   
        n_jobs_permutation,
        keep_permutation_statistics,
        verbose
    ))
}


#' Print a summary of the multivariate test results.
#' 
#' @param kt output of [`rktest::ktest()`] function.
#' @param long boolean (default = `FALSE`). If `TRUE`, the print is more 
#' detailed. 
#' @param t integer (default = `10`), if `long == FALSE`, value for truncation 
#' parameter to be considered.
#' @param ts vector of integer (default = `c(1,5,10)`), if `long == FALSE`, 
#' values for truncation parameter to be considered.
#'
#' @return No return.
#' @export
#'
#' @examples
#' # WRITE ME
print_multivariate_test_results <- function(
        kt, long = FALSE, t = NULL, ts = c(1,5,10)
) {
    # check input
    assert_logical(long, len = 1)
    assert_count(t, null.ok = TRUE)
    assert_integerish(ts, null.ok = TRUE)
    
    # Python call
    return(kt$print_multivariate_test_results(long, t, ts))
}

#' Get p-values table with respect to the preset truncation parameter.
#' 
#' @param kt output of [`rktest::ktest()`] function.
#' @param contrib boolean, if `TRUE`, returns the p-value associated to 
#' each principal component individually, otherwise returns the p-value 
#' associated to the `kfda` statistic. 
#' @param log boolean, if `TRUE`, returns the log p-values.
#' @param verbose integer (`1` or `0`) or logical (`TRUE` or `FALSE`), 
#' enable/disable verbosity.
#'
#' @return FIXME
#' @export
#'
#' @examples
#' # WRITE ME
get_pvalue <- function(
        kt,
        contrib=False,
        log=False,
        verbose=0
) {
    return(kt$get_pvalue(contrib = contrib, log = log, verbose = verbose))
}