test_that("install_ktest", {
    skip_if_no_python()
    
    expect_error(install_ktest(), NA)
})
