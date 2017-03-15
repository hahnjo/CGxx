find_package(OpenMP)

if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

  add_executable(cg_omp $<TARGET_OBJECTS:common>
    CGOpenMP.cpp
  )
endif()
