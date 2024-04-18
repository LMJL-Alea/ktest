test_that("check_python", {
    checkmate::expect_flag(check_python(verbose = FALSE))
    skip_if_no_python()
    expect_true(expect_message(check_python()))
    expect_false(expect_warning(check_python(version = "100000")))
})

test_that("check_ktest", {
    skip_if_no_python()
    checkmate::expect_flag(check_ktest(verbose = FALSE))
})
