if ("${CMAKE_C_COMPILER_ID}" MATCHES "Clang|XL")
  return()
endif()

find_package(CUDA)

if (CUDA_FOUND)
  if (${CMAKE_MAJOR_VERSION} LESS 3 OR (${CMAKE_MAJOR_VERSION} EQUAL 3 AND ${CMAKE_MINOR_VERSION} LESS 3))
    # CMake 3.3.0 and later will handle this flag internally!
    set(CUDA_NVCC_FLAGS "-std=c++11")
  endif()

  cuda_add_executable(cg_cuda $<TARGET_OBJECTS:common>
    CGCUDABase.cu
    CGCUDA.cu
    kernel.cu
  )

  cuda_add_executable(cg_multi_cuda $<TARGET_OBJECTS:common>
    CGCUDABase.cu
    CGMultiCUDA.cu
    kernel.cu
  )
endif()
