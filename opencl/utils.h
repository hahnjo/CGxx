// SPDX-License-Identifier:	GPL-3.0-or-later

#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdlib>
#include <iostream>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"

static inline void checkError(cl_int error) {
  if (error != CL_SUCCESS) {
    std::cerr << "OpenCL error: " << error << std::endl;
    std::exit(1);
  }
}

static inline void checkedSetKernelArg(cl_kernel kernel, cl_uint arg_index,
                                       size_t arg_size, const void *arg_value) {
  checkError(clSetKernelArg(kernel, arg_index, arg_size, arg_value));
}

static inline void checkedReleaseMemObject(cl_mem buffer) {
  checkError(clReleaseMemObject(buffer));
}

static inline int calculateGroups(int N, int local, int maxGroups) {
  int groups, div = 1;

  // We have grid-stride loops so it should be better if all groups receive
  // roughly the same amount of work.
  do {
    groups = std::ceil(((double)N) / local / div);
    div++;
  } while (groups > maxGroups);

  return groups;
}

#endif
