include(CheckCXXCompilerFlag)

check_cxx_compiler_flag(-std=c++11 CGXX_HAVE_STD_CPP11_FLAG)
check_cxx_compiler_flag(-Wall CGXX_HAVE_WALL_FLAG)

check_cxx_compiler_flag(-pthread CGXX_HAVE_PTHREAD_FLAG)
