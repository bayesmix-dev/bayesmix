% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{read_many_proto_from_file}
\alias{read_many_proto_from_file}
\title{Read many protobuf messages of the same type from a file}
\usage{
read_many_proto_from_file(filename, msg_type)
}
\value{
A list of \code{RProtoBuf::Message} of type \code{msg_type}
}
\description{
This function parse the file given by \code{filename} and deserialize all
protobuf messages of type \code{msg_type}. The latter is of type
\code{RProtoBuf::Descriptor}
}
