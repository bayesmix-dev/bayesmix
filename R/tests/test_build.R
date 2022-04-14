# Test for build_bayesmix function
test_build <- function() {
  success = bayesmixr::build_bayesmix()
  stopifnot(success)
}
