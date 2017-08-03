set(OpenCL_INCLUDE_DIR ".")
if (DEFINED ENV{CUDA_ROOT})
  set(OpenCL_LIBRARY "$ENV{CUDA_ROOT}/lib64/libOpenCL.so")
elseif (DEFINED ENV{POCL_ROOT})
  set(OpenCL_LIBRARY "$ENV{POCL_ROOT}/lib64/libOpenCL.so")
endif()
find_package(OpenCL)

if (OpenCL_FOUND)
  set(OPENCL_SVM_AVAILABLE OFF)
  # clSVMAlloc and clSVMFree has been introduced with OpenCL 2.0.
  if (OpenCL_VERSION_MAJOR GREATER 1)
    message(STATUS "SVM allocation available with OpenCL")
    set(OPENCL_SVM_AVAILABLE ON)
  endif()
  option(OPENCL_USE_SVM "Use SVM allocation with OpenCL." ${OPENCL_SVM_AVAILABLE})

  if (OPENCL_USE_SVM AND NOT OPENCL_SVM_AVAILABLE)
    message(WARNING "SVM allocation NOT available, forcing off!")
    set(OPENCL_USE_SVM OFF)
  endif()

  configure_file(clSVM.h.in clSVM.h)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})


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
