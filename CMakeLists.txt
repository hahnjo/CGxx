cmake_minimum_required(VERSION 2.8)

project(CGxx C CXX)
if (${CMAKE_CXX_FLAGS} MATCHES "-qhalt=e")
  string(REPLACE "-qhalt=e" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

# Get compiler features.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(config-ix)
# We want to use C++11!
if (NOT CGXX_HAVE_STD_CPP11_FLAG)
  message(FATAL_ERROR "Compiler does not support C++11!")
endif()
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

if (CGXX_HAVE_WALL_FLAG)
  set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  option(GCC_OFFLOADING "Use offloading with the GNU Compiler Collection" OFF)
endif()

add_library(common OBJECT
  CG.cpp
  Matrix.cpp
  Preconditioner.cpp
  WorkDistribution.cpp
)

add_subdirectory(cuda)
add_subdirectory(openacc)
add_subdirectory(opencl)
add_subdirectory(openmp)
add_subdirectory(serial)
