test_that("saving-loading_ktest", {
    
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
    # run default test
    test(kt_1, stat = 'kfda', permutation = FALSE, verbose = 0)
    
    
    # saving/loading Ktest object
    withr::with_tempfile("temp_file_for_saving", {
        # saving
        save_ktest(kt_1, str_c(temp_file_for_saving, ".gz"), compress=TRUE)
        
        # loading
        kt_2 <- load_ktest(str_c(temp_file_for_saving, ".gz"), compressed=TRUE)
        
        # check
        expect_equal(get_pvalues(kt_1), get_pvalues(kt_2))
    })
    
})
