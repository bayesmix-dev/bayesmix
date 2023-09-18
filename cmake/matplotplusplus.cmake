# Define patch command to inject
set(matplotplusplus_patch git apply ${BASEPATH}/resources/patches/matplotplusplus.patch)

# Make matplotplusplus available (+ patch)
message(STATUS "")
message(STATUS "Fetching alandefreitas/matplotplusplus")
FetchContent_Declare(matplotplusplus
	DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	PATCH_COMMAND ${matplotplusplus_patch}
	URL "https://github.com/alandefreitas/matplotplusplus/archive/refs/tags/v1.2.0.tar.gz"
)
FetchContent_MakeAvailable(matplotplusplus)
