library("bayesmixr")

test_build <- function() {
  success = build_bayesmix()
  stopifnot(success)
}
