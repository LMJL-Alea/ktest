#' Saving and loading `ktest` results
#' @name saving-loading
#' @rdname saving-loading
#' @aliases save_ktest
#' @author Ghislain Durif
#' 
#' @description
#' Save a computed `Ktest` objects or a list of computed `Ktest` objects 
#' into a binary file, and load it back into R.
#' 
#' @details
#' Since `ktest` R package is basically a wrapper around the `ktest` Python 
#' package, we manipulate Python objects directly in R:
#' ```{r class, eval=FALSE}
#' class(kt_1)
#' ## [1] "ktest.tester.Ktest"   "ktest.kernel_statistics.Statistics"
#' ## [3] "python.builtin.object"
#' ```
#' Such type of objects cannot be saved and loaded using the standard 
#' [base::save()] and [base::load()] functions in R.
#' 
#' To overcome this issue, the `save_ktest()` and `load_ktest()` functions 
#' are wrapper around Python `Ktest` object `save()` and `load()` methods,
#' using a particular pickling (a.k.a. serializing) framework
#' (provided by the `dill` Python package).
#' 
#' Note: `.pkl` or `.pickle` file extensions are considered the "standard" 
#' extension for such file storing serialized Python objects but you may use
#' any extension that suits you.
#' 
#' When enabling compression, the input object will be saved under the
#' following filename `<filename>.gz` since Gzip compression is used
#' internally.
#'
#' @param obj a `Ktest` object to be saved.
#' @param filename string, name of the file to save input object or to load
#' from.
#' @param compress boolean flag to enable/disable compression when saving Ktest 
#' object to disk.
#'
#' @return no return
#'
#' @importFrom checkmate assert_class assert_string assert_path_for_output
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # data loading
#' tmp <- load_example_data()
#' # gene expression data table (344 cells and 83 genes)
#' data_tab <- tmp$data_tab
#' # metadata table with sampling conditions (for the 344 cells)
#' metadata_tab <- tmp$metadata_tab
#' 
#' # create Ktest object
#' kt_1 = ktest_init(
#'     data = data_tab, metadata = metadata_tab, 
#'     sample_names = c('0H', '48HREV')
#' )
#' # save test result
#' save_ktest(kt_1, filename = "kt_1.pkl", compress=TRUE)
#' # load test result
#' kt_1 <- load_ktest(filename = "kt_1.pkl.gz", compressed=TRUE)
#' }
save_ktest <- function(obj, filename, compress = TRUE) {
    # check input
    assert_class(obj, "ktest.tester.Ktest")
    assert_path_for_output(filename)
    # saving
    obj$save(filename, compress)
}

#' @rdname saving-loading
#'
#' @param compressed boolean flag to enable/disable decompression when loading
#' Ktest object from disk.
#' @return the loaded object that was saved in `filename` file.
#' 
#' @importFrom checkmate assert_file
#'
#' @export
load_ktest <- function(filename, compressed=TRUE) {
    # check input
    assert_file(filename)
    # file loading
    return(pyktest$tester$Ktest$load(filename, compressed))
}
