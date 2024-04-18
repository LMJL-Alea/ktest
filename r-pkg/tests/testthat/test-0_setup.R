test_that("skip_if_no_python", {
    # Test that a skip happens
    if(reticulate::py_available()) {
        # no skip
        expect_condition(skip_if_no_python(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_no_python(), class = "skip")
    }
})

test_that("skip_if_no_pyktest", {
    # Test that a skip happens
    if(reticulate::py_module_available("pyktest")) {
        # no skip
        expect_condition(skip_if_no_pyktest(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_no_pyktest(), class = "skip")
    }
})

test_that("skip_if_not_interactive", {
    # Test that a skip happens
    if(!interactive()) {
        # no skip
        expect_condition(skip_if_not_interactive(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_not_interactive(), class = "skip")
    }
})
