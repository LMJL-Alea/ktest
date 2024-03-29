% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/python.R
\name{check_ktest}
\alias{check_ktest}
\title{Check if \code{ktest} package is ready to run.}
\usage{
check_ktest(warn = TRUE)
}
\arguments{
\item{warn}{boolean, if TRUE (default), warn user about check result.}
}
\value{
boolean value indicating if the \code{ktest} package is ready.
}
\description{
In practice, check if \code{ktest} Python package is installed (which is done
by the function \code{\link[=install_ktest]{install_ktest()}}).
}
\examples{
check_ktest()
}
\seealso{
\code{\link[=install_ktest]{install_ktest()}}
}
