include(CheckCXXSourceCompiles)
set(SRC "
int main() {
  double a[4];
  #pragma omp target enter data map(alloc: a[0:4])
#pragma omp target map(a[0:4])
  { }
  return 0;
}
")

set(_OPENMP_REQUIRED_VARS)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
  set(OPENMP_TARGET_ACCELERATOR
    nvptx64-nvidia-cuda
    CACHE
    STRING
    "The target to use for OpenMP target directives"
  )
  set(OPENMP_TARGET_FLAGS "-fopenmp -fopenmp-targets=${OPENMP_TARGET_ACCELERATOR}")

  find_package(CUDA QUIET)
  if (CUDA_FOUND)
    set(OPENMP_TARGET_FLAGS
      "${OPENMP_TARGET_FLAGS} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
  endif()
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND GCC_OFFLOADING)
  set(OPENMP_TARGET_FLAGS "-fopenmp")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
  set(OPENMP_TARGET_FLAGS "-qopenmp")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "XL")
  set(OPENMP_TARGET_FLAGS "-qsmp=omp -qoffload")
endif ()

if (OPENMP_TARGET_FLAGS)
  set(CMAKE_REQUIRED_FLAGS "${OPENMP_TARGET_FLAGS}")
  check_cxx_source_compiles("${SRC}" CompilesOpenMPTarget)
  if (CompilesOpenMPTarget)
    list(APPEND _OPENMP_REQUIRED_VARS OPENMP_TARGET_FLAGS)
  endif ()
endif ()

if (_OPENMP_REQUIRED_VARS)
  include(FindPackageHandleStandardArgs)

  find_package_handle_standard_args(OpenMPTarget
                                    REQUIRED_VARS ${_OPENMP_REQUIRED_VARS})

  mark_as_advanced(${_OPENMP_REQUIRED_VARS})

  unset(_OPENMP_REQUIRED_VARS)
endif ()
