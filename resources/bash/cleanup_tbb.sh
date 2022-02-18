#!/usr/bin/env bash

# Run this script if build_tbb.py gives a "busy file" error at line 45.

# Folders from build_tbb.py
# stan_math_lib=lib/math/lib
# tbb_root=lib/math/lib/tbb_2019_U8
# tbb_debug=lib/math/lib/tbb_debug
# tbb_release=lib/math/lib/tbb_release
# tbb_dir=lib/math/lib/tbb
rm -rf lib/math/lib/tbb_debug/
mv lib/math/lib/tbb_2019_U8/include/ lib/math/lib/tbb
rm -rf lib/math/lib/tbb_2019_U8/
for name in lib/math/lib/tbb_release/*; do
  mv $name lib/math/lib/tbb
done
rm -rf lib/math/lib/tbb_release
