CGxx - Object-Oriented Implementation of the Conjugate Gradients Method
=======================================================================

This implementation can make use of different programming models:
 * CUDA
 * OpenACC
 * OpenCL
 * OpenMP

If applicable, the offloading programming model also includes a version for multiple devices.
In addition, there is a serial implementation for reference.

Environment variables
---------------------

| Name | Description | Allowed values | Default value |
| --- | --- | --- | --- |
| `CG_MAX_ITER` | Maximum number of iterations | integer greater than zero | 1000 |
| `CG_TOLERANCE` | Tolerance for convergence | number greater than zero | 1e-9 |
| `CG_MATRIX_FORMAT` | Matrix format to use in computation | `COO`, `CRS`, `ELL` | depends on programming model |
| `CG_PRECONDITIONER` | Preconditioner to use | `none`, `jacobi` | depends on programming model |
| `CG_WORK_DISTRIBUTION` | Way of distributing work to multiple devices | `row`, `nz` | `row` |
| `CG_OVERLAPPED_GATHER` | Whether to overlap computation and communication for multiple devices | `0` = disabled | depends on programming model |
| `CG_CUDA_GATHER_IMPL` | Implementation to use for gathering in `matvec` kernel | `host`, `device`, `p2p` | `host` |
| `CG_OCL_PARALLEL_TRANSFER_TO` | Whether to transfer the data to the device in parallel | `0` = disabled | enabled |
| `CG_OCL_GATHER_IMPL` | Implementation to use for gathering in `matvec` kernel | `host`, `device` | `host` |

License
-------

The code is released under the GNU General Public License v3:

    Copyright (C) 2017  Jonas Hahnfeld

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
