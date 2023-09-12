#' Return a decoder for a basic varint value (does not include tag).
#'
#' Decoded values will be bitwise-anded with the given mask before being
#' returned, e.g. to limit them to 32 bits.  The returned decoder does not take
#' the usual "end" parameter -- the caller is expected to do bounds checking
#' after the fact (often the caller can defer such checking until later). The
#' decoder returns a (value, new_pos) pair.
#'
#' @keywords internal
VarintDecoder = function(mask, result_type) {

  # Define DecodeVarint function
  DecodeVarint <- function(buffer, pos) {
    result = 0
    shift = 0
    while (TRUE) {
      b = as.numeric(buffer[pos])
      result = bitops::bitOr(result, bitops::bitShiftL(bitops::bitAnd(b, 0x7f), shift))
      # result <- bitops::bitOr(result, bitops::bitAnd(b, 0x7f)*(2L^shift))
      pos = pos + 1
      if (!bitops::bitAnd(b, 0x80)) {
        result <- bitops::bitAnd(result, mask)
        result <- result_type(result)
        return(list(result = result, pos = as.integer(pos)))
      }
      shift <- shift + 7
      if (shift >= 64) {
        stop('Too many bytes when decoding varint.')
      }
    }
  }

  # Return the decoder as result
  return(DecodeVarint)
}

#' Use this decoder version for values which must be limited to 32 bits.
#'
#' @keywords internal
DecodeVarint32 = VarintDecoder(2^32 - 1, as.integer)
