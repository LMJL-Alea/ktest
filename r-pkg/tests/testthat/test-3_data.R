test_that("load_example_data", {
    out <- load_example_data()
    checkmate::expect_list(out, len = 2)
    checkmate::expect_subset(names(out), c("data_tab", "metadata_tab"))
    checkmate::expect_data_frame(
        out$data_tab, nrows = 344, ncols = 83, types = "numeric")
    checkmate::expect_data_frame(
        out$metadata_tab, nrows = 344, ncols = 1, types = "character")
})
