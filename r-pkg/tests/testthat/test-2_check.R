test_that("check_python", {
    checkmate::expect_flag(check_python(verbose = FALSE))
    skip_if_no_python()
    expect_true(check_python(verbose = FALSE))
    expect_false(check_python(verbose = FALSE, version = "100000"))
})

test_that("check_ktest", {
    skip_if_no_python()
    checkmate::expect_flag(check_ktest(verbose = FALSE))
})
