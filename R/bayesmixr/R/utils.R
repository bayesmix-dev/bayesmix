#' Print a protobuf message to file only if input is not a file
#'
#' If \code{maybe_proto} is a file, returns the file name. If \code{maybe_proto}
#' is a string representing a message, prints the message to a file and returns
#' the file name.
#'
#' @keywords internal
maybe_print_to_file <- function(maybe_proto, proto_name = NULL, out_dir = NULL) {
  if(file.exists(maybe_proto)){
    return(maybe_proto)
  }
  proto_file = sprintf("%s/%s.asciipb", out_dir, proto_name)
  write(maybe_proto, file = proto_file)
  return(proto_file)
}

#' Import Protocol Buffers Descriptors of bayesmix
#'
#' This utility loads in the workspace the protocol buffers descriptors defined
#' in the \code{bayesmix} library, via \code{RProtoBuf} package. These
#' descriptors can be used to handle the MCMC chain output of
#' \code{\link{run_mcmc}} function.
#'
#' @return NULL
#'
#' @export
import_protobuf_messages <- function() {

  # Get bayesmixr.Renviron file and parse it
  renviron <- system.file("bayesmixr.Renviron", package = "bayesmixr")
  readRenviron(renviron)

  # Deduce BUILD_DIR from RSPATIALCMC_EXE variable in RspatialCMC.Renviron
  BAYESMIX_HOME <- dirname(dirname(Sys.getenv("BAYESMIX_EXE")))

  # Deduce protocol buffer proto paths
  bayesmix_protoPath <- sprintf("%s/src/proto", BAYESMIX_HOME)

  # Import protocol buffer message descripts via RProtoBuf
  RProtoBuf::readProtoFiles2(protoPath = bayesmix_protoPath)
}


#' Read many protobuf messages of the same type from a file
#'
#' This function parse the file given by \code{filename} and deserialize all
#' protobuf messages of type \code{msg_type}. The latter is of type
#' \code{RProtoBuf::Descriptor}
#'
#' @return A list of \code{RProtoBuf::Message} of type \code{msg_type}
#'
#' @keywords internal
read_many_from_file <- function(filename, msg_type) {

  # Open the binary file for reading
  connection <- file(filename, "rb")
  buffer <- readBin(connection, "raw", file.size(filename))
  close(connection)

  # Prepare output list
  out <- list()

  # Parse the file and deserialize messages
  n = 1
  while (n < length(buffer)) {

    # Decode varint and get message length and new position
    decoder_res = bayesmixr:::DecodeVarint32(buffer, n)
    msg_len = decoder_res$result
    new_pos = decoder_res$pos

    # Prepare single message buffer
    n = new_pos
    msg_buf = buffer[n:(n+msg_len-1)]

    # Deserialize message and update counters
    tryCatch(
      {
        out = append(out, RProtoBuf::read(msg_type, msg_buf))
        n = n + msg_len
      },
      error = function(e){
        message("Something went wrong while deserialization: ")
        message(e)
        out <- NULL
        break
      })
  }
  # Return chain
  return(out)
}

# Usage example
# import_protobuf_messages()
#
# file_path <- "resources/tutorial/out/chains.recordio"
# msg_type <- bayesmix.AlgorithmState
#
# chain <- read_many_from_file(file_path, msg_type)
