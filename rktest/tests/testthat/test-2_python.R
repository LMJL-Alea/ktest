# FIXME
# test_that("check_python", {
#     skip_if_no_python()
#     skip_if_not_interactive()
#     expect_true(check_python())
#     expect_warning(check_python(version = "100000"))
#     expect_false(check_python(version = "100000", warn = FALSE))
# })
# 
# test_that("check_pyktest", {
#     skip_if_no_python()
#     skip_if_not_interactive()
#     
#     run_py_env({
#         cat(str(reticulate::py_discover_config()))
#         expect_warning(check_pyktest())
#         expect_false(check_pyktest(warn = FALSE))
#     })
#     
#     run_py_env({
#         install_pyktest()
#         expect_true(check_pyktest())
#     })
# })
