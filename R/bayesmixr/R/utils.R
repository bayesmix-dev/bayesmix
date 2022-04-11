# If maybe_proto is a file, returns the file name. If maybe_proto is a string representing a message,
# prints the message to a file and returns the file name.
maybe_print_to_file <- function(maybe_proto, proto_name = NULL, out_dir = NULL) {
  if(file.exists(maybe_proto)){
    return(maybe_proto)
  }
  proto_file = sprintf("%s/%s.asciipb", out_dir, proto_name)
  write(maybe_proto, file = proto_file)
  return(proto_file)
}
