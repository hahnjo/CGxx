set(OpenCL_INCLUDE_DIR ".")
if (DEFINED ENV{CUDA_ROOT})
  set(OpenCL_LIBRARY "$ENV{CUDA_ROOT}/lib64/libOpenCL.so")
elseif (DEFINED ENV{POCL_ROOT})
  set(OpenCL_LIBRARY "$ENV{POCL_ROOT}/lib64/libOpenCL.so")
endif()
find_package(OpenCL)

if (OpenCL_FOUND)
  # https://cmake.org/pipermail/cmake/2014-September/058647.html
  add_library(OpenCL UNKNOWN IMPORTED)
  set_property(TARGET OpenCL PROPERTY IMPORTED_LOCATION "${OpenCL_LIBRARIES}")

  add_executable(cg_ocl $<TARGET_OBJECTS:common>
    CGOpenCLBase.cpp
    CGOpenCL.cpp
  )
  target_link_libraries(cg_ocl OpenCL)

  add_executable(cg_multi_ocl $<TARGET_OBJECTS:common>
    CGOpenCLBase.cpp
    CGMultiOpenCL.cpp
  )
  target_link_libraries(cg_multi_ocl OpenCL)
endif()
