# Define patch command to inject
set(math-tbb_patch git apply ${BASEPATH}/resources/patches/math-tbb.patch)

# Fetching bayesmix-dev/math
message(STATUS "")
message(STATUS "Fetching bayesmix-dev/math")
FetchContent_Declare(math
  GIT_REPOSITORY "https://github.com/bayesmix-dev/math.git"
  GIT_TAG "develop"
  PATCH_COMMAND ${math-tbb_patch}
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

# Add TBB link directory
link_directories(${TBB_ROOT})

# In Windows, write absolute path in ~/.bash_profile
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  file(APPEND $ENV{HOME}/.bash_profile "PATH=${TBB_ROOT}:$PATH\n")
endif()

# Find math packages
# find_package(math REQUIRED)
