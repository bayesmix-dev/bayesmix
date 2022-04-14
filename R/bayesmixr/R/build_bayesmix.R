#' Builds the BayesMix executable
#'
#' After the build, if no error has occurred, it saves the path into the \code{BAYESMIX_EXE} environment variable.
#' Such variable is defined only when this package is loaded in the R session.
#'
#' @param nproc Number of processes to use for parallel compilation. Thanks to \code{parallel} package,
#' this parameter defaults to half of the available processes (through \code{\link[parallel]{detectCores}} function)
#' @param build_dirname Name for the build directory of BayesMix. Default is "build".
#' @return \code{TRUE} if build is successfull, \code{FALSE} otherwise
#'
#' @export
build_bayesmix <- function(nproc = ceiling(parallel::detectCores()/2), build_dirname = "build") {

  # Check input types
  if(!is.numeric(nproc)) { stop("nproc must be a number") }
  if(!is.character(build_dirname)) { stop("build_dirname must be a string") }

  # Set bayesmix_home folder from BAYESMIXR_HOME
  home_dir = Sys.getenv("BAYESMIXR_HOME")
  bayesmix_home = dirname(dirname(home_dir))

  # Build bayesmix
  build_dir = sprintf("%s/%s", bayesmix_home, build_dirname)
  flags = "-DDISABLE_DOCS=TRUE -DDISABLE_BENCHMARKS=TRUE -DDISABLE_TESTS=TRUE -DDISABLE_PLOTS=TRUE -DCMAKE_BUILD_TYPE=Release"
  cat("Configuring bayesmix ... ")
  CONFIGURE = sprintf("mkdir -p %s && cd %s && cmake .. %s", build_dir, build_dir, flags)
  tryCatch({
    system(CONFIGURE, ignore.stdout = TRUE, ignore.stderr = TRUE)
    cat("DONE ✔ \n")
    },
    warning = function(w){
      message(sprintf("\nWarning during configure: %s\n", as.character(w)))
    },
    error = function(e) {
      message(sprintf("\nConfigure failed with error: %s\n)", as.character(e)))
      return(FALSE)
    })

  BUILD = sprintf("cd %s && make run_mcmc -j%d", build_dir, nproc)
  tryCatch({
    # Make run_mcmc executable
    cat("Building bayesmix executable:\n")
    system(BUILD)

    # Set BAYESMIX_EXE environment variable for R
    renviron = system.file(".Renviron", package = "bayesmixr")
    if(Sys.getenv("BAYESMIX_EXE") == ""){
      cat("Setting BAYESMIX_EXE environment variable ... ")
      write(x = sprintf("BAYESMIX_EXE=%s/run_mcmc", build_dir), file = renviron, append = TRUE)
      readRenviron(renviron)
      cat("DONE ✔ \n")
    }
    cat("Successfully installed bayesmix.\n")
    return(TRUE)
  },
  warning = function(w){
    message(sprintf("\nWarning during build: %s\n", as.character(w)))
  },
  error = function(e){
    message(sprintf("\nBuild failed with error: %s\n)", as.character(e)))
    return(FALSE)
  })
}
