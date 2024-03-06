# Define patch command to inject
set(matplotplusplus_patch git apply ${BASEPATH}/resources/patches/matplotplusplus.patch)

# Make matplotplusplus available (+ patch)
message(STATUS "")
message(STATUS "Fetching alandefreitas/matplotplusplus")
FetchContent_Declare(matplotplusplus
	GIT_REPOSITORY "https://github.com/alandefreitas/matplotplusplus.git"
	GIT_TAG "v1.2.1"
	PATCH_COMMAND ${matplotplusplus_patch}
)
FetchContent_MakeAvailable(matplotplusplus)
