/*
    Copyright (C) 2017  Jonas Hahnfeld

    This file is part of CGxx.

    CGxx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CGxx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CGxx.  If not, see <http://www.gnu.org/licenses/>. */

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
