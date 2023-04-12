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
