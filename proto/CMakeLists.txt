
file(GLOB PROTOBUF_DEFINITION_FILES "*.proto")

set(PROTOBUF_INPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/proto")
set(PROTOBUF_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/proto/cpp/")

foreach( proto_file ${PROTOBUF_DEFINITION_FILES} )
   get_filename_component(proto_name ${proto_file} NAME_WE)
   set_source_files_properties("${PROTOBUF_OUTPUT_DIRECTORY}/${proto_name}.pb.cc"
                               "${PROTOBUF_OUTPUT_DIRECTORY}/${proto_name}.pb.h"
                               PROPERTIES GENERATED TRUE)
   message("${proto_file}")
   message("${PROTOBUF_OUTPUT_DIRECTORY}/${proto_name}.pb.cc}")
   message("${PROTOBUF_OUTPUT_DIRECTORY}/${proto_name}.pb.h")
endforeach()


foreach(file ${PROTOBUF_DEFINITION_FILES})
    set(PROTOBUF_ARGUMENTS "protoc –proto_path=\"${PROTOBUF_INPUT_DIRECTORY}\" –cpp_out=\"${PROTOBUF_OUTPUT_DIRECTORY}\" \"${file}\"")
    execute_process(COMMAND ${PROTOBUF_OUTPUT_DIRECTORY}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE PROTOBUF_RESULT
            OUTPUT_VARIABLE PROTOBUF_OUTPUT_VARIABLE)
endforeach()

file(GLOB PROTOBUF_MODELS_INCLUDES "proto/cpp/*.pb.cc" "proto/cpp/*.hpp")

target_sources(bayesmix
  PUBLIC
    "${PROTOBUF_MODELS_INCLUDES}"
)
