# Fetching bayesmix-dev/math
message(STATUS "")
message(STATUS "Fetching bayesmix-dev/math")
FetchContent_Declare(math
  GIT_REPOSITORY "https://github.com/bayesmix-dev/math.git"
  GIT_TAG "develop"
)
FetchContent_MakeAvailable(math)

# Set TBB_ROOT variable
set(TBB_ROOT ${math_SOURCE_DIR}/lib/tbb)

# Define make command
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(MAKE_COMMAND mingw32-make)
else()
  set(MAKE_COMMAND make)
endif()

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

# Add TBB link directory
link_directories(${TBB_ROOT})

# In Windows, add TBB_ROOT to PATH variable via batch file if not present
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # Check if adding TBB_ROOT is already present in PATH
  file(TO_CMAKE_PATH "$ENV{PATH}" PATH)
  string(FIND "${PATH}" "${TBB_ROOT}" tbb_path-LOCATION)
  # If not present, add to PATH user environment variable
  if(tbb_path-LOCATION EQUAL -1)
    execute_process(
      COMMAND cmd.exe /C install-tbb.bat
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${BASEPATH}
    )
    if(result)
      message(FATAL_ERROR "Failed to install TBB (${result})!")
    endif()
  endif()
endif()
