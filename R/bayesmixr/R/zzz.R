# Parse internal renviron file to set BAYESMIX_EXE variable
.onAttach <- function(...) {
  readRenviron(system.file("bayesmixr.Renviron", package = "bayesmixr"))
}

# Unset BAYESMIX_EXE variable on detaching
.onDetach <- function(...) {
  Sys.unsetenv("BAYESMIXR_HOME")
  Sys.unsetenv("BAYESMIX_EXE")
  Sys.unsetenv("TBB_PATH")
}
