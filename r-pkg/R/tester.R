#' Generate a Ktest object that will be used to compute the specified 
#' sample kernel-test-based comparison
#' 
#' @description
#' Ktest object can be used to perform kernel tests, such as maximal mean 
#' discrepancy test (MMD) and a test based on kernel Fisher Discriminant 
#' Analysis (kFDA).
#' 
#' @details
#' See 'ktest' package vignette for more details and usage examples. You can 
#' also refer to Ozier-Lafontaine et al (2024) for more details about the 
#' methods.
#' 
#' @references
#' Ozier-Lafontaine A., Fourneaux C., Durif G., Arsenteva P., Vallot C., 
#' Gandrillon O., Gonin-Giraud S., Michel B., Picard F. (2024). 
#' Kernel-Based Testing for Single-Cell Differential Analysis. Preprint. 
#' \doi{10.48550/arXiv.2307.08509};
#' [arXiv.2307.08509](https://arxiv.org/abs/2307.08509);
#' [hal-04214858](https://hal.science/hal-04214858).
#' 
#' @param data `matrix` or `data.frame`.
#'     The data to test, containing two samples at the same time. 
#'     Rows correspond to observations/samples, and columns correspond to 
#'     features.
#' 
#' @param metadata `matrix` or `data.frame` with a single column.
#'     The metadata associated with the data to test. The values should 
#'     indicate the labels of the samples in `'data'`. At the moment, the
#'     library can only perform a two sample test, thus two or more levels 
#'     are expected but only the first two detected levels are used, or the
#'     levels corresponding to the labels given in `samples_names` 
#'     (if provided). The first dimensions of `'data'` and `'metadata'`,
#'     corresponding to observations, should coincide.
#'     
#' @param sample_names NULL or `vector`.
#'     If given, should contain exactly two strings, corresponding to
#'     labels that should be selected for testing in the `metadata` table. 
#'     If NULL, sample names are derived from the metadata label levels 
#'     (the first two detected levels are used).
#' 
#' @param kernel_function character string or callable (function).
#'     Specifies the kernel function. Acceptable values in the form of a
#'     string are `'gauss'` (default) and `'linear'`. Pass a callable 
#'     for a user-defined kernel function.
#' 
#' @param kernel_bandwidth character string or numeric.
#'     Value of the bandwidth for kernels using a bandwidth. If `'median'` 
#'     (default), the bandwidth will be set as the median or its multiple, 
#'     depending on the value of the parameter `kernel_median_coef`. Pass a 
#'     numeric for a user-defined value of the bandwidth.
#' 
#' @param kernel_median_coef numeric.
#'     Multiple of the median to compute bandwidth if 
#'     `kernel_bandwidth = 'median'`. The default is 1.
#'     
#' @param nystrom logical.
#'     If TRUE, computes the Nystrom approximation, and uses it to compute 
#'     the test statistics. The default if FALSE.
#'     
#' @param n_landmarks integer.
#'     Number of landmarks used in the Nystrom method. If unspecified, one
#'     fifth of the observations are selected as landmarks.
#'     
#' @param n_anchors integer.
#'     Number of anchors used in the Nystrom method, by default equal to
#'     the number of landmarks.
#'     
#' @param landmark_method `'random'` or `'kmeans++'`.
#'     Method of the landmarks selection, `'random'` (default) corresponds 
#'     to selecting landmarks among the observations according to the 
#'     random uniform distribution.
#'     
#' @param anchor_basis character string.
#'     Options for different ways of computing the covariance operator of 
#'     the landmarks in the Nystrom method, of which the anchors are the 
#'     eigenvalues. Possible values are `'w'` (default),`'s'` and `'k'`.
#'
#' @param random_state integer or NULL.
#'     Determines random number generation for the Nystrom approximation
#'     and for the permutations. If NULL (default), the generator is 
#'     the RandomState instance used by Numpy (i.e. `np.random`) in Python. 
#'     To ensure the results are reproducible, pass an integer to 
#'     instanciate the seed, or a Python RandomState instance (recommended).
#'
#' @return a `Ktest` object.
#' @export
#' 
#' @seealso [ktest::test()], [ktest::get_pvalues()], [ktest::get_statistics()], 
#' [ktest::get_proj()]
#' 
#' @importFrom checkmate 
#' assert 
#' assert_character 
#' assert_choice 
#' assert_count 
#' assert_flag 
#' assert_number 
#' check_choice 
#' check_data_frame 
#' check_function 
#' check_matrix 
#' check_number
#'
#' @examples
#' \dontrun{
#' # data loading
#' tmp <- load_example_data()
#' # gene expression data table (344 cells and 83 genes)
#' data_tab <- tmp$data_tab
#' # metadata table with sampling conditions (for the 344 cells)
#' metadata_tab <- tmp$metadata_tab
#' 
#' # create Ktest object
#' kt_1 = ktest_init(
#'     data = data_tab, metadata = metadata_tab, 
#'     sample_names = c('0H', '48HREV')
#' )
#' print(kt_1)
#' }
ktest_init <- function(
        data, metadata, 
        sample_names = NULL,
        kernel_function = 'gauss',
        kernel_bandwidth = 'median',
        kernel_median_coef = 1, 
        nystrom = FALSE, 
        n_landmarks = NULL, 
        landmark_method = 'random',
        n_anchors = NULL, 
        anchor_basis = 'w', 
        random_state = NULL
) {
    # check input
    assert(check_data_frame(data), check_matrix(data))
    assert(
        check_data_frame(metadata, min.cols = 1), 
        check_matrix(metadata, ncols = 1))
    assert_character(sample_names, len = 2, null.ok = TRUE)
    assert(
        check_choice(kernel_function, c("gauss", "linear")), 
        check_function(kernel_function))
    assert(
        check_choice(kernel_bandwidth, "median"), 
        check_number(kernel_bandwidth))
    assert_number(kernel_median_coef)
    assert_flag(nystrom)
    assert_count(n_landmarks, null.ok = TRUE)
    assert_choice(landmark_method, c("random", "kmeans++"))
    assert_count(n_anchors, null.ok = TRUE)
    assert_choice(anchor_basis, c("w", "s", "k"))
    assert_count(random_state, null.ok = TRUE)
    
    # type cast
    if(!is.null(n_landmarks)) n_landmarks <- as.integer(n_landmarks)
    if(!is.null(n_anchors)) n_anchors <- as.integer(n_anchors)
    if(!is.null(random_state)) random_state <- as.integer(random_state)
    
    # Python call
    return(pyktest$tester$Ktest(
        data, metadata, 
        sample_names,
        kernel_function,
        kernel_bandwidth,
        kernel_median_coef, 
        nystrom, 
        n_landmarks, 
        landmark_method,
        n_anchors, 
        anchor_basis, 
        random_state
    ))
}

#' Compute multivariate testing
#' 
#' @description
#' Performs either the MMD or the kFDA test to compare the two samples of 
#' the considered dataset. Stores the results in the corresponding attribute 
#' depending on the statistic and on the p-value approach.
#' 
#' @details
#' See 'ktest' package vignette for more details and usage examples. You can 
#' also refer to Ozier-Lafontaine et al (2024) for more details about the 
#' methods.
#' 
#' @references
#' Ozier-Lafontaine A., Fourneaux C., Durif G., Arsenteva P., Vallot C., 
#' Gandrillon O., Gonin-Giraud S., Michel B., Picard F. (2024). 
#' Kernel-Based Testing for Single-Cell Differential Analysis. Preprint. 
#' \doi{10.48550/arXiv.2307.08509};
#' [arXiv.2307.08509](https://arxiv.org/abs/2307.08509);
#' [hal-04214858](https://hal.science/hal-04214858).
#' 
#' @param kt a `Ktest` object, output of [`ktest::ktest_init()`] function.
#' @param stat character string
#'     The test statistic to use, can be either `'kfda'` (default) or `'mmd'`. 
#' @param permutation logical
#'     If `FALSE` (default), the asymptotic approach is applied to compute 
#'     p-values. If `TRUE`, a permutation test is performed. If `stat = "mmd"` 
#'     then permutation based p-value computation is always used and 
#'     `permutation` value is ignored.
#' @param n_permutations integer
#'     Number of permutations performed. The default is 500.
#' @param verbose integer
#'     The higher the verbosity, the more messages keeping track of 
#'     computations. The default is 1.
#'     - < 1: no messages,
#'     - 1: progress bar with computation time,
#'     - 2: warnings are printed once,
#'     - 3: warnings are printed every time they appear.
#' 
#' @return No return. Input Ktest object `kt` is updated with test results.
#' @export
#' 
#' @seealso [ktest::ktest_init()], [ktest::get_pvalues()], 
#' [ktest::get_statistics()], [ktest::get_proj()]
#' 
#' @importFrom checkmate
#' assert_choice
#' assert_class
#' assert_count
#' assert_flag
#' 
#' @examples
#' \dontrun{
#' # data loading
#' tmp <- load_example_data()
#' # gene expression data table (344 cells and 83 genes)
#' data_tab <- tmp$data_tab
#' # metadata table with sampling conditions (for the 344 cells)
#' metadata_tab <- tmp$metadata_tab
#' 
#' # create Ktest object
#' kt_1 = ktest_init(
#'     data = data_tab, metadata = metadata_tab, 
#'     sample_names = c('0H', '48HREV')
#' )
#' # run test
#' test(kt_1)
#' # print results
#' print(kt_1)
#' }
test <- function(
        kt, 
        stat = 'kfda', 
        permutation = FALSE, 
        n_permutations = 500, 
        verbose = 1
) {
    # check input
    assert_class(kt, c("ktest.tester.Ktest"))
    assert_choice(stat, c("kfda", "mmd"))
    assert_flag(permutation)
    assert_count(n_permutations, positive = TRUE)
    assert_count(verbose)
    
    # Python call
    kt$test(
        stat,
        permutation,
        as.integer(n_permutations),
        as.integer(verbose)
    )
}


#' Kernel test p-values
#' 
#' @description
#' Get p-values table with respect to the preset truncation parameter.
#' 
#' @details
#' See Ozier-Lafontaine et al (2024) for more details.
#' 
#' @references
#' Ozier-Lafontaine A., Fourneaux C., Durif G., Arsenteva P., Vallot C., 
#' Gandrillon O., Gonin-Giraud S., Michel B., Picard F. (2024). 
#' Kernel-Based Testing for Single-Cell Differential Analysis. Preprint. 
#' \doi{10.48550/arXiv.2307.08509};
#' [arXiv.2307.08509](https://arxiv.org/abs/2307.08509);
#' [hal-04214858](https://hal.science/hal-04214858).
#' 
#' @inheritParams test
#' @param t_max integer, the maximum truncation value to consider, i.e. the 
#' maximum dimension for the kernel embedding projections. Default is NULL and 
#' the truncation value will be the number of observations in the data.
#' 
#' @return a vector of p-values for each dimension up to the truncation value
#' or `NULL` value if corresponding `stat` was not computed.
#' @export
#' 
#' @seealso [ktest::ktest_init()], [ktest::test()], [ktest::get_statistics()], 
#' [ktest::get_proj()]
#' 
#' @importFrom checkmate
#' assert_choice
#' assert_class
#' assert_count
#' assert_flag
#'
#' @examples
#' \dontrun{
#' # data loading
#' tmp <- load_example_data()
#' # gene expression data table (344 cells and 83 genes)
#' data_tab <- tmp$data_tab
#' # metadata table with sampling conditions (for the 344 cells)
#' metadata_tab <- tmp$metadata_tab
#' 
#' # create Ktest object
#' kt_1 = ktest_init(
#'     data = data_tab, metadata = metadata_tab, 
#'     sample_names = c('0H', '48HREV')
#' )
#' # run test
#' test(kt_1)
#' # get p-values
#' get_pvalues(kt_1)
#' }
get_pvalues <- function(kt, stat = 'kfda', permutation = FALSE, t_max = NULL) {
    # check input
    assert_class(kt, c("ktest.tester.Ktest"))
    assert_choice(stat, c("kfda", "mmd"))
    assert_flag(permutation)
    assert_count(t_max, positive = TRUE, null.ok = TRUE)
    
    # init output
    out <- NULL
    
    # assert which p-values to return
    if((stat == "kfda")) {
        if(permutation) {
            out <- kt$kfda_pval_perm
        } else {
            out <- kt$kfda_pval_asymp
        }
    } else {
        out <- kt$mmd_pval_perm
    }
    
    # post-process kFDA test
    if(!is.null(out) && (stat == "kfda")) {
        # truncation
        if(!is.null(t_max)) out <- out[1:min(t_max, length(out))]
        # rename output
        names(out) <- as.character(1:length(out))
    }
    
    # output
    return(out)
}

#' Kernel test statistic
#' 
#' @description
#' Get statistic table with respect to the preset truncation parameter.
#' 
#' @details
#' See Ozier-Lafontaine et al (2024) for more details.
#' 
#' @references
#' Ozier-Lafontaine A., Fourneaux C., Durif G., Arsenteva P., Vallot C., 
#' Gandrillon O., Gonin-Giraud S., Michel B., Picard F. (2024). 
#' Kernel-Based Testing for Single-Cell Differential Analysis. Preprint. 
#' \doi{10.48550/arXiv.2307.08509};
#' [arXiv.2307.08509](https://arxiv.org/abs/2307.08509);
#' [hal-04214858](https://hal.science/hal-04214858).
#' 
#' @inheritParams get_pvalues
#' @param contrib logical, works only with `stat = 'kfda'`, if TRUE, 
#' returns the vector of statistic contributions to each direction, 
#' otherwise returns the cumulative statistic values with regards to the 
#' direction. Ignored if `stat = 'mmd'`.
#'
#' @return a vector of statistic values for each dimension up to 
#' the truncation or `NULL` value if corresponding `stat` was not computed.
#' @export
#' 
#' @seealso [ktest::ktest_init()], [ktest::test()], [ktest::get_pvalues()], 
#' [ktest::get_proj()]
#' 
#' @importFrom checkmate
#' assert_choice
#' assert_class
#' assert_flag
#'
#' @examples
#' # WRITE ME
get_statistics <- function(kt, stat = 'kfda', contrib = FALSE, t_max = NULL) {
    # check input
    assert_class(kt, c("ktest.tester.Ktest"))
    assert_choice(stat, c("kfda", "mmd"))
    assert_flag(contrib)
    assert_count(t_max, positive = TRUE, null.ok = TRUE)
    
    # init output
    out <- NULL
    
    # assert which statistic to return
    if((stat == "kfda")) {
        if(contrib) {
            out <- kt$kfda_statistic_contrib
        } else {
            out <- kt$kfda_statistic
        }
    } else {
        out <- kt$mmd_statistic
    }
    
    # post-process kFDA test
    if(!is.null(out) && (stat == "kfda")) {
        # truncation
        if(!is.null(t_max)) out <- out[1:min(t_max, length(out))]
        # rename output
        names(out) <- as.character(1:length(out))
    }
    
    # output
    return(out)
}

#' Kernel projections
#' 
#' @encoding UTF-8
#' @description
#' Get kernel projections for each condition with respect to the preset 
#' truncation parameter for kernel FDA (kfda) test.
#' 
#' @details
#' See Ozier-Lafontaine et al (2024) for more details.
#' 
#' @references
#' Ozier-Lafontaine A., Fourneaux C., Durif G., Arsenteva P., Vallot C., 
#' Gandrillon O., Gonin-Giraud S., Michel B., Picard F. (2024). 
#' Kernel-Based Testing for Single-Cell Differential Analysis. Preprint. 
#' \doi{10.48550/arXiv.2307.08509};
#' [arXiv.2307.08509](https://arxiv.org/abs/2307.08509);
#' [hal-04214858](https://hal.science/hal-04214858).
#' 
#' @inheritParams test
#' @inheritParams get_pvalues
#' @param contrib logical. If FALSE (default), a list of discriminant 
#' axis projections per directions for each condition in the sample is 
#' returned. If TRUE, a list of contributions to the discriminant 
#' axis projections per directions for each condition in the sample is 
#' returned.
#'
#' @return a list containing, for each sample condition, a projection matrix 
#' or projection contribution matrix stored in a `data.frame` of dimension 
#' the number of observations in the conditions × the number of dimensions up 
#' to the truncation.
#' @export
#' 
#' @importFrom checkmate
#' assert_class
#' assert_flag
#' assert_count
#'
#' @examples
#' # WRITE ME
get_proj <- function(kt, contrib = FALSE, t_max = NULL) {
    # check input
    assert_class(kt, c("ktest.tester.Ktest"))
    assert_flag(contrib)
    assert_count(t_max, positive = TRUE, null.ok = TRUE)
    
    # truncation
    if(is.null(t_max)) t_max <- kt$data$ntot
    
    # compute projections
    kt$project(as.integer(t_max))
    
    # init output
    out <- NULL
    
    # assert projection to return
    if(contrib) {
        out <- kt$kfda_proj_contrib
    } else {
        out <- kt$kfda_proj
    }
    
    # output
    return(out)
}
