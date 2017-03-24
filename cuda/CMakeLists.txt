find_package(CUDA)

if (CUDA_FOUND)
  # For use with older versions of CMake
  set(CUDA_NVCC_FLAGS "-std=c++11;-lineinfo")

  cuda_add_executable(cg_cuda $<TARGET_OBJECTS:common>
    CGCUDA.cu
    kernel.cu
  )
endif()
