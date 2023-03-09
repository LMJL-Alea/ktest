# test_that("install_pyktest", {
#     skip_if_no_python()
#     # work in dedicated environment
#     envname <- random_envname("ktest", length = 30)
#     # run pyktest install
#     run_py_env({
#         # install pyktest
#         install_pyktest(method = "virtualenv", envname = envname)
#         # print python config
#         pyconf <- reticulate::py_discover_config()
#         cat(str(pyconf))
#         # verification
#         expect_true(reticulate::py_module_available("pyktest"))
#     }, envname)
# })
