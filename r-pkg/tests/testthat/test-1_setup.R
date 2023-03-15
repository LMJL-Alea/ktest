test_that("random_envname", {
    
    expect_equal(random_envname(prefix = "", length = 0), "")
    
    res <- random_envname(prefix = "", length = 10)
    checkmate::expect_string(res, n.char = 10, pattern = "[A-Za-z0-9]{10}")
    
    prefix <- "prefix"
    res <- random_envname(prefix, length = 10)
    checkmate::expect_string(
        res, n.char = 10 + stringr::str_length(prefix),
        pattern = stringr::str_c("^", prefix, "[A-Za-z0-9]{10}")
    )
})

# FIXME
# test_that("run_py_env", {
#     
#     skip_if_not_interactive()
#     
#     # test env init
#     expect_no_error(run_py_env({
#         have_python <- reticulate::py_available()
#         expect_true(have_python)
#     }))
#     
#     # specific environment name
#     envname <- random_envname("ktest", length = 20)
#     # test specific environment init
#     run_py_env({
#         pyconf <- reticulate::py_discover_config()
#         expect_true(
#             pyconf$python,
#             file.path(reticulate::virtualenv_root(), envname, "bin", "python")
#         )
#     }, envname)
# })

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
