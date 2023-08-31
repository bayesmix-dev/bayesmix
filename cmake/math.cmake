# Define patch command to inject
set(math-tbb_patch git apply ${BASEPATH}/resources/patches/math-tbb.patch)

# Fetching bayesmix-dev/math
message(STATUS "")
message(STATUS "Fetching bayesmix-dev/math")
FetchContent_Declare(math
  GIT_REPOSITORY "https://github.com/bayesmix-dev/math.git"
  GIT_TAG "develop"
  PATCH_COMMAND ${math-tbb_patch}
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(math)

# Set TBB_ROOT variable
set(TBB_ROOT ${math_SOURCE_DIR}/lib/tbb)

# Define make command
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(MAKE_COMMAND mingw32-make)
else()
  set(MAKE_COMMAND make)
endif()

# Set compiler flags
file(APPEND ${math_SOURCE_DIR}/make/local "TBB_CXX_TYPE=gcc\n")

# Set extra compiler flags for Windows
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  file(APPEND ${math_SOURCE_DIR}/make/local "CXXFLAGS+=-Wno-nonnull\n")
  file(APPEND ${math_SOURCE_DIR}/make/local "TBB_CXXFLAGS=-U__MSVCRT_VERSION__ -D__MSVCRT_VERSION__=0x0E00\n")
endif()

# Compile math libraries
message(STATUS "Compiling math libraries ...")
execute_process(
  COMMAND ${MAKE_COMMAND} -f ./make/standalone math-libs
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${math_SOURCE_DIR}
)
if(result)
  message(FATAL_ERROR "Failed to compile math libraries (${result})!")
endif()

# Generate .lib equivalents if in Windows
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  # Generating .def files for each .dll file
  file(GLOB DllFiles "${TBB_ROOT}/*.dll")
  foreach(DLL_FILE IN LISTS DllFiles)
    # Get def and lib files path
    get_filename_component(DLL_NAME ${DLL_FILE} NAME_WE)
    message("DLL_NAME: ${DLL_NAME}")
    # set(DEF_FILE ${TBB_ROOT}/${DLL_NAME}-new.def)
    # message("DEF_FILE: ${DEF_FILE}")
    # set(LIB_FILE ${TBB_ROOT}/${DLL_NAME}.lib)
    # message("LIB_FILE: ${LIB_FILE}")
    # # Generating .def files
    # message(STATUS "Generating .def file for ${DLL_FILE}:")
    # execute_process(
    #   COMMAND gendef - ${DLL_FILE}
    #   RESULT_VARIABLE result
    #   OUTPUT_FILE ${DEF_FILE}
    #   WORKING_DIRECTORY ${TBB_ROOT}
    # )
    # if(result)
    #   message(FATAL_ERROR "Failed to generate .def file for ${DLL_FILE}: (${result})!")
    # endif()
    # # Generating .lib files
    # message(STATUS "Generating .lib file for ${DEF_FILE}:")
    # execute_process(
    #   COMMAND dlltool -d ${DEF_FILE} -l ${LIB_FILE}
    #   RESULT_VARIABLE result
    #   WORKING_DIRECTORY ${TBB_ROOT}
    # )
    # if(result)
    #   message(FATAL_ERROR "Failed to generate .lib file for ${DEF_FILE}: (${result})!")
    # endif()
    # # Check if current library can be found
    # message("")
    # find_library(CURR_LIB ${DLL_NAME} HINTS ${TBB_ROOT} REQUIRED)
    # # list(APPEND TBB_LIB_LIST "${CURR_LIB}")
    # unset(CURR_LIB CACHE)
  endforeach()
endif()

# Generate TBB_LIBRARIES variable
# list(JOIN TBB_LIB_LIST " " TBB_LIBRARIES)
# find_library(TBB tbb HINTS ${TBB_ROOT} REQUIRED)
# message("TBB_LIBRARIES: ${TBB_LIBRARIES}")

# Add TBB link directory
link_directories(${TBB_ROOT})

# Find math packages
find_package(math REQUIRED)
