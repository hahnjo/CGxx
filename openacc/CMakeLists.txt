find_package(OpenACC)

if (OPENACC_FOUND)
  set(CMAKE_CXX_FLAGS "${OPENACC_FLAGS} ${CMAKE_CXX_FLAGS}")

  add_executable(cg_acc $<TARGET_OBJECTS:common>
    CGOpenACC.cpp
  )
endif()
