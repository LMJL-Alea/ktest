test_that("skip_if_no_python", {
    # Test that a skip happens
    if(!reticulate::py_available()) {
        expect_condition(skip_if_no_python(), class = "skip")
    } else {
        # avoid empty test
        expect_true(TRUE)
    }
})

test_that("skip_if_no_pyktest", {
    # Test that a skip happens
    if(!reticulate::py_module_available("ktest")) {
        expect_condition(skip_if_no_pyktest(), class = "skip")
    } else {
        # avoid empty test
        expect_true(TRUE)
    }
})
