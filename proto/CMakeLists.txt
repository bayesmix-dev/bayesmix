FILE(GLOB HDRS "cpp/*.pb.h")
FILE(GLOB SRCS "cpp/*.pb.cc")

target_sources(bayesmix
  PUBLIC
    ${HDRS}
    ${SRCS}
)
