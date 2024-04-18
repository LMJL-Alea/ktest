test_that("ktest_init", {
    
    skip_if_no_python()
    skip_if_no_pyktest()
    
    # data loading
    tmp <- load_example_data()
    # gene expression data table (344 cells and 83 genes)
    data_tab <- tmp$data_tab
    # metadata table with sampling conditions (for the 344 cells)
    metadata_tab <- tmp$metadata_tab

    # create Ktest object
    kt_1 = ktest_init(
        data = data_tab, metadata = metadata_tab,
        sample_names = c('0H', '48HREV')
    )
    
    # check
    checkmate::expect_class(
        kt_1, 
        c("ktest.tester.Ktest", "ktest.kernel_statistics.Statistics", 
          "python.builtin.object")
    )
})

test_that("test", {
    
    # data loading
    tmp <- load_example_data()
    # gene expression data table (344 cells and 83 genes)
    data_tab <- tmp$data_tab
    # metadata table with sampling conditions (for the 344 cells)
    metadata_tab <- tmp$metadata_tab
    
    # create Ktest object
    kt_1 = ktest_init(
        data = data_tab, metadata = metadata_tab,
        sample_names = c('0H', '48HREV')
    )
    
    # check
    expect_null(kt_1$kfda_statistic)
    expect_null(kt_1$kfda_pval_asymp)
    expect_null(kt_1$kfda_pval_perm)
    expect_null(kt_1$mmd_statistic)
    expect_null(kt_1$mmd_pval_perm)
    
    # run default test
    test(kt_1, stat = 'kfda', permutation = FALSE, verbose = 0)
    
    # check
    checkmate::expect_numeric(kt_1$kfda_statistic, min.len = 1)
    checkmate::expect_numeric(kt_1$kfda_pval_asymp, min.len = 1)
    expect_null(kt_1$kfda_pval_perm)
    expect_null(kt_1$mmd_statistic)
    expect_null(kt_1$mmd_pval_perm)
    
    # run kfda test with permutation
    test(
        kt_1, stat = 'kfda', permutation = TRUE, n_permutations = 10, 
        verbose = 0)
    
    # check
    checkmate::expect_numeric(kt_1$kfda_pval_perm, min.len = 1)
    expect_null(kt_1$mmd_statistic)
    
    # run mmd test without permutation
    test(kt_1, stat = 'mmd', n_permutations = 10, verbose = 0)
    
    # check
    checkmate::expect_numeric(kt_1$mmd_statistic, min.len = 1)
    checkmate::expect_numeric(kt_1$mmd_pval_perm, min.len = 1)
})

test_that("get_pvalues", {
    # data loading
    tmp <- load_example_data()
    # gene expression data table (344 cells and 83 genes)
    data_tab <- tmp$data_tab
    # metadata table with sampling conditions (for the 344 cells)
    metadata_tab <- tmp$metadata_tab
    
    # create Ktest object
    kt_1 = ktest_init(
        data = data_tab, metadata = metadata_tab,
        sample_names = c('0H', '48HREV')
    )
    
    # check
    expect_null(get_pvalues(kt_1, stat = 'kfda', permutation = FALSE))
    expect_null(get_pvalues(kt_1, stat = 'kfda', permutation = TRUE))
    expect_null(get_pvalues(kt_1, stat = 'mmd'))
    
    # run default test
    test(kt_1, stat = 'kfda', permutation = FALSE, verbose = 0)
    
    # truncation param
    t_max <- 100
    
    # check
    res <- get_pvalues(kt_1, stat = 'kfda', permutation = FALSE, t_max = t_max)
    checkmate::expect_numeric(res, len = t_max)
    expect_equal(names(res), as.character(1:length(res)))
    expect_equal(
        unname(res),
        unname(kt_1$kfda_pval_asymp)[1:t_max], tolerance = 1e-8
    )
    
    expect_null(get_pvalues(kt_1, stat = 'kfda', permutation = TRUE))
    expect_null(get_pvalues(kt_1, stat = 'mmd'))
    
    # run kfda test with permutation
    test(
        kt_1, stat = 'kfda', permutation = TRUE, n_permutations = 10, 
        verbose = 0)
    
    # check
    res <- get_pvalues(kt_1, stat = 'kfda', permutation = TRUE, t_max = t_max)
    checkmate::expect_numeric(res, len = t_max)
    expect_equal(names(res), as.character(1:length(res)))
    expect_equal(
        unname(res),
        unname(kt_1$kfda_pval_perm)[1:t_max], tolerance = 1e-8
    )
    
    expect_null(get_pvalues(kt_1, stat = 'mmd'))
    
    # run mmd test without permutation
    test(kt_1, stat = 'mmd', n_permutations = 10, verbose = 0)
    
    # check
    res <- get_pvalues(kt_1, stat = 'mmd')
    checkmate::expect_number(res)
    expect_equal(res, kt_1$mmd_pval_perm, tolerance = 1e-8)
})

test_that("get_statistics", {
    # data loading
    tmp <- load_example_data()
    # gene expression data table (344 cells and 83 genes)
    data_tab <- tmp$data_tab
    # metadata table with sampling conditions (for the 344 cells)
    metadata_tab <- tmp$metadata_tab
    
    # create Ktest object
    kt_1 = ktest_init(
        data = data_tab, metadata = metadata_tab,
        sample_names = c('0H', '48HREV')
    )
    
    # check
    expect_null(get_statistics(kt_1, stat = 'kfda', contrib = FALSE))
    expect_null(get_statistics(kt_1, stat = 'kfda', contrib = TRUE))
    expect_null(get_statistics(kt_1, stat = 'mmd'))
    
    # run default test
    test(kt_1, stat = 'kfda', permutation = FALSE, verbose = 0)
    
    # truncation param
    t_max <- 100
    
    # check
    res <- get_statistics(kt_1, stat = 'kfda', contrib = FALSE, t_max = t_max)
    checkmate::expect_numeric(res, len = t_max)
    expect_equal(names(res), as.character(1:length(res)))
    expect_equal(
        unname(res),
        unname(kt_1$kfda_statistic)[1:t_max], tolerance = 1e-8
    )
    
    res <- get_statistics(kt_1, stat = 'kfda', contrib = TRUE, t_max = t_max)
    checkmate::expect_numeric(res, len = t_max)
    expect_equal(names(res), as.character(1:length(res)))
    expect_equal(
        unname(res),
        unname(kt_1$kfda_statistic_contrib)[1:t_max], tolerance = 1e-8
    )
    
    expect_null(get_statistics(kt_1, stat = 'mmd'))
    
    # run mmd test without permutation
    test(kt_1, stat = 'mmd', n_permutations = 10, verbose = 0)
    
    # check
    res <- get_statistics(kt_1, stat = 'mmd')
    checkmate::expect_number(res)
    expect_equal(res, kt_1$mmd_statistic, tolerance = 1e-8)
})

test_that("get_proj", {
    # data loading
    tmp <- load_example_data()
    # gene expression data table (344 cells and 83 genes)
    data_tab <- tmp$data_tab
    # metadata table with sampling conditions (for the 344 cells)
    metadata_tab <- tmp$metadata_tab
    
    # create Ktest object
    kt_1 = ktest_init(
        data = data_tab, metadata = metadata_tab,
        sample_names = c('0H', '48HREV')
    )
    
    # truncation param
    t_max <- 100
    
    # projection
    res <- get_proj(kt_1, contrib = FALSE, t_max = t_max)
    
    # check
    expected_nrow <- unname(table(metadata_tab$condition))
    
    checkmate::expect_list(res, len = 2)
    checkmate::expect_set_equal(names(res), unique(metadata_tab$condition))
    
    for(i in 1:2) {
        checkmate::expect_data_frame(
            res[[i]], types = "numeric", 
            nrows = expected_nrow[i], ncols = t_max
        )
    }
    
    # projection contrib
    res <- get_proj(kt_1, contrib = TRUE, t_max = t_max)
    
    # check
    expected_nrow <- unname(table(metadata_tab$condition))
    
    checkmate::expect_list(res, len = 2)
    checkmate::expect_set_equal(names(res), unique(metadata_tab$condition))
    
    for(i in 1:2) {
        checkmate::expect_data_frame(
            res[[i]], types = "numeric", 
            nrows = expected_nrow[i], ncols = t_max
        )
    }
})
