find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

# Find all the public headers
get_target_property(BAYESMIX_PUBLIC_HEADER_DIR bayesmix INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE BAYESMIX_PUBLIC_HEADERS ${PROJECT_SOURCE_DIR}/src/*.h)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Doxygen won't create this for us
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# Only regenerate Doxygen when the Doxyfile or public headers change
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${BAYESMIX_PUBLIC_HEADERS}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating docs"
                   VERBATIM)

# Nice named target so we can run the job easily
add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

find_program(protoc-plugin protoc-gen-doc)
message("protoc-plugin: " ${protoc-plugin})
if (protoc-plugin)
  add_custom_target(document_protos
    COMMAND protobuf::protoc -I${PROTO_DIR}
    --plugin=protoc-gen-doc=${protoc-plugin}
    --doc_out=${CMAKE_CURRENT_LIST_DIR}
    --doc_opt=html,protos.html
    ${PROTO_DIR}/*.proto
  )
else()
  find_program(Docker_EXECUTABLE docker)
  if(NOT Docker_EXECUTABLE)
      message(FATAL_ERROR "Cannot find the docker executable!")
  endif()
  add_custom_target(document_protos
      COMMAND /usr/local/bin/docker run --platform linux/amd64 --rm
        -v ${BASEPATH}/docs:/out -v ${PROTO_DIR}:/protos
        pseudomuto/protoc-gen-doc -I${PROTO_DIR} --doc_opt=html,protos.html
  )
endif()

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND
                     ${SPHINX_EXECUTABLE} -b html
                     # Tell Breathe where to find the Doxygen output
                     -Dbreathe_projects.bayesmix=${DOXYGEN_OUTPUT_DIR}/xml
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
                   # Other docs files you want to track should go here (or in some variable)
                   ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                   ${DOXYGEN_INDEX_FILE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/protos.html
                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                   COMMENT "Generating documentation with Sphinx")

# Nice named target so we can run the job easily
add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

# Add an install target to install the docs
include(GNUInstallDirs)
install(DIRECTORY ${SPHINX_BUILD}
  DESTINATION ${CMAKE_INSTALL_DOCDIR})

add_custom_target(document_bayesmix)
add_dependencies(document_bayesmix document_protos)
add_dependencies(document_bayesmix Doxygen)
add_dependencies(document_bayesmix Sphinx)
