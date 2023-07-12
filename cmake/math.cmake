# Fetching bayesmix-dev/math
message(STATUS "")
message(STATUS "Fetching bayesmix-dev/math")
FetchContent_Declare(math
  GIT_REPOSITORY "https://github.com/bayesmix-dev/math.git"
  GIT_TAG "develop"
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(math)

# Build bayesmix-dev/math TBB library 
message(STATUS "Build bayesmix-dev/math TBB")

# Define TBB_ROOT Folder
set(TBB_ROOT ${math_SOURCE_DIR}/lib/tbb)
file(COPY ${math_SOURCE_DIR}/lib/tbb_2020.3/ DESTINATION ${TBB_ROOT})

# Build TBB Library with CMake Integration
include(${TBB_ROOT}/cmake/TBBBuild.cmake)
list(APPEND MAKE_ARGS "tbb_build_dir=${TBB_ROOT}")
list(APPEND MAKE_ARGS "tbb_build_prefix=tbb")
tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS ${MAKE_ARGS})

# Find math and TBB packages
find_package(math REQUIRED)
find_package(TBB REQUIRED)