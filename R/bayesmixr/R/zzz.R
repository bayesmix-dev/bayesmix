# Parse internal renviron file to set BAYESMIX_EXE variable
.onAttach <- function(...) {
  readRenviron(system.file(".Renviron", package = "bayesmixr"))
}

# Unset BAYESMIX_EXE variable on detaching
.onDetach <- function(...) {
  Sys.unsetenv("BAYESMIXR_HOME")
  Sys.unsetenv("BAYESMIX_EXE")
}
