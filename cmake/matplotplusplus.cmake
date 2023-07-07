# Define patch command to inject
set(matplotplusplus_patch git apply ${BASEPATH}/resources/patches/matplotplusplus.patch)

# Make matplotplusplus available (+ patch)
message(STATUS "Fetching matplotplusplus")
FetchContent_Declare(matplotplusplus
	GIT_REPOSITORY "https://github.com/alandefreitas/matplotplusplus"
	GIT_TAG "origin/master"
	PATCH_COMMAND ${matplotplusplus_patch}
	OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(matplotplusplus)

# Find matplotplusplus
find_package(matplotplusplus REQUIRED)