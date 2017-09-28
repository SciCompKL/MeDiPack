/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2017 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, TU Kaiserslautern)
 *
 * This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * MeDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * MeDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with MeDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum (SciComp, TU Kaiserslautern)
 */

#pragma once

#define MEDI_MPI_VERSION_1_0 100
#define MEDI_MPI_VERSION_1_1 101
#define MEDI_MPI_VERSION_1_2 102
#define MEDI_MPI_VERSION_1_3 103
#define MEDI_MPI_VERSION_2_0 200
#define MEDI_MPI_VERSION_2_1 201
#define MEDI_MPI_VERSION_2_2 202
#define MEDI_MPI_VERSION_3_0 300
#define MEDI_MPI_VERSION_3_1 301

#ifndef MEDI_MPI_TARGET
# define MEDI_MPI_TARGET MEDI_MPI_VERSION_3_1
#endif

namespace medi {
  #define MEDI_UNUSED(name) (void)(name)
  #define MEDI_CHECK_ERROR(expr) (expr)
}
