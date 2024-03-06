# CMake function that add compilation instructions for every .proto files in
# a given FOLDER, passed as input.

function(compile_protobuf_files)
    # Parse input arguments
    set(oneValueArgs FOLDER HEADERS SOURCES PYTHON_OUT_PATH)
    set(multiValueArgs INCLUDE_PROTO_PATHS)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Append all paths for protoc
    list(APPEND PROTO_DIRS "--proto_path=${arg_FOLDER}")
    if(NOT "${arg_INCLUDE_PROTO_PATHS}" STREQUAL "")
        foreach(PBPATH IN LISTS arg_INCLUDE_PROTO_PATHS)
            list(APPEND PROTO_DIRS "--proto_path=${PBPATH}")
        endforeach()
    endif()

    # Set --python-out option if PYTHON_OUT is set
    if(NOT "${arg_PYTHON_OUT_PATH}" STREQUAL "")
        set(PYTHON_OUT "--python_out=${arg_PYTHON_OUT_PATH}")
    endif()

    # Make custom command to compile each ProtoFile in FOLDER_PATH
    file(GLOB ProtoFiles "${arg_FOLDER}/*.proto")
    set(PROTO_DIR proto)
    foreach(PROTO_FILE IN LISTS ProtoFiles)
    message(STATUS "protoc proto(cc): ${PROTO_FILE}")
    get_filename_component(PROTO_DIR ${PROTO_FILE} DIRECTORY)
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    set(PROTO_HDR ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.h)
    set(PROTO_SRC ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.cc)
    message(STATUS "protoc hdr: ${PROTO_HDR}")
    message(STATUS "protoc src: ${PROTO_SRC}")
    add_custom_command(
        OUTPUT ${PROTO_SRC} ${PROTO_HDR}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE} ${PROTO_DIRS}
        "--cpp_out=${PROJECT_BINARY_DIR}" ${PYTHON_OUT}
        ${PROTO_FILE}
        DEPENDS ${PROTO_FILE} ${Protobuf_PROTOC_EXECUTABLE}
        COMMENT "Generate C++ protocol buffer for ${PROTO_FILE}"
        VERBATIM)
    list(APPEND PROTO_HEADERS ${PROTO_HDR})
    list(APPEND PROTO_SOURCES ${PROTO_SRC})
    endforeach()
    SET_SOURCE_FILES_PROPERTIES(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

    # Propagate PROTO_HDRS and PROTO_SRCS to parent scope
    set(${arg_HEADERS} ${PROTO_HEADERS} PARENT_SCOPE)
    set(${arg_SOURCES} ${PROTO_SOURCES} PARENT_SCOPE)
endfunction()
