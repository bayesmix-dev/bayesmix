# Test for build_bayesmix function
test_build <- function() {
  readRenviron(system.file(".Renviron", package = "bayesmixr"))
  stopifnot(bayesmixr::build_bayesmix())
}
