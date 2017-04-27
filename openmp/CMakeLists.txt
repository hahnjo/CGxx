find_package(OpenMP)

if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

  add_executable(cg_omp $<TARGET_OBJECTS:common>
    CGOpenMP.cpp
  )
endif()


find_package(OpenMPTarget)

if (OPENMPTARGET_FOUND)
  set(CMAKE_CXX_FLAGS "${OPENMP_TARGET_FLAGS} ${CMAKE_CXX_FLAGS}")

  add_executable(cg_omp_target $<TARGET_OBJECTS:common>
    CGOpenMPTarget.cpp
  )

  add_executable(cg_multi_omp_target $<TARGET_OBJECTS:common>
    CGMultiOpenMPTarget.cpp
  )
endif()
