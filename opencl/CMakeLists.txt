set(OpenCL_INCLUDE_DIR ".")
if (DEFINED ENV{CUDA_ROOT})
  set(OpenCL_LIBRARY "$ENV{CUDA_ROOT}/lib64/libOpenCL.so")
endif()
find_package(OpenCL)

if (OpenCL_FOUND)
  add_executable(cg_ocl $<TARGET_OBJECTS:common>
    CGOpenCLBase.cpp
    CGOpenCL.cpp
  )
  target_link_libraries(cg_ocl ${OpenCL_LIBRARIES})
endif()
