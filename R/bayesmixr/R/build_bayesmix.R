#' Builds the BayesMix executable
#'
#' After the build, if no error has occurred, it saves the path into the \code{BAYESMIX_EXE} environment variable.
#' Such variable is defined only when this package is loaded in the R session.
#'
#' @param nproc Number of processes to use for parallel compilation. Thanks to \code{parallel} package,
#' this parameter defaults to half of the available processes (through \code{\link[parallel]{detectCores}} function)
#' @param build_subdir Name for the sub-directory of \code{bayesmix/} folder in which configuration and compilation happens.
#' Default value is \code{build}.
#' @return No output if build is successful, it raises errors otherwise
#'
#' @export
build_bayesmix <- function(nproc = ceiling(parallel::detectCores()/2), build_subdir = "build") {

  # Check input types
  if(!is.numeric(nproc)) { stop("nproc must be a number") }
  if(!is.character(build_subdir)) { stop("build_subdir must be a string") }

  # Get .Renviron file from package
  renviron = system.file("bayesmixr.Renviron", package = "bayesmixr")

  # Set bayesmix_home folder from BAYESMIXR_HOME
  readRenviron(renviron)
  home_dir = Sys.getenv("BAYESMIXR_HOME")
  if(home_dir == ""){
    stop("Something went wrong while installing bayesmixr R package.")
  }
  bayesmix_home = dirname(dirname(home_dir))

  # Create build/ subdirectory
  build_dir = sprintf("%s/%s", bayesmix_home, build_subdir)
  dir.create(build_dir, showWarnings = F)

  # Configure bayesmix
  cat("*** Configuring BayesMix ***\n")
  flags = '-DDISABLE_TESTS=TRUE -DDISABLE_PLOTS=TRUE -DCMAKE_BUILD_TYPE=Release'
  CONFIGURE = sprintf('cmake .. -G "Unix Makefiles" %s', flags)
  errlog <- withr::with_dir(build_dir, system(CONFIGURE, ignore.stderr = TRUE))
  if(errlog != 0L){
    errmsg <- "Something went wrong during configure: command '%s' exit with status %d"
    stop(sprintf(errmsg, CONFIGURE, errlog))
  }
  cat("\n")

  # Build bayesmix::run_mcmc executable
  cat("*** Building BayesMix executable ***\n")
  BUILD = sprintf('make run_mcmc -j%d', nproc)
  errlog <- withr::with_dir(build_dir, system(BUILD))
  if (errlog != 0L) {
    errmsg <- "Something went wrong during build: command '%s' exit with status %d"
    stop(sprintf(errmsg, BUILD, errlog))
  }
  cat("\n")

  # Set BAYESMIX_EXE environment variable
  cat("*** Setting BAYESMIX_EXE environment variable ***\n")
  write(x = sprintf('BAYESMIX_EXE=%s/run_mcmc', build_dir), file = renviron, append = TRUE)
  cat("\n")

  # Set TBB_PATH environment variable
  cat("*** Setting TBB_PATH environment variable ***\n")
  tbb_path = sprintf('%s/lib/_deps/math-src/lib/tbb', bayesmix_home)
  write(x = sprintf('TBB_PATH=%s', tbb_path), file = renviron, append = TRUE)
  cat("\n")

  # Parse .Renviron file to get environment variables
  readRenviron(renviron)
  cat("Successfully installed BayesMix\n")
}
