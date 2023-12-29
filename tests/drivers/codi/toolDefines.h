/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * MeDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * MeDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License for more details.
 * You should have received a copy of the GNU
 * Lesser General Public License along with MeDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum, Tim Albring (SciComp, University of Kaiserslautern-Landau)
 */

#pragma once

#include <codi.hpp>
#include <medi/medi.hpp>
#if CODI_MAJOR_VERSION >= 2
  #include <codi/tools/mpi/codiMpiTypes.hpp>
#else
  #include <codi/externals/codiMpiTypes.hpp>
#endif

typedef CODI_TYPE NUMBER;

#ifndef VECTOR
# define VECTOR 0
#endif

#ifndef UNTYPED
# define UNTYPED 0
#endif

#ifndef FORWARD_TAPE
# define FORWARD_TAPE 0
#endif

#ifndef PRIMAL_TAPE
# define PRIMAL_TAPE 0
#endif

#if CODI_MAJOR_VERSION >= 2
  #define TOOL_TYPE codi::CoDiMpiTypes<NUMBER>
#else
  #define TOOL_TYPE CoDiMpiTypes<NUMBER>
#endif
#define TOOL codiTypes

extern TOOL_TYPE* codiTypes;

#include "../globalDefines.h"

#if UNTYPED
  #undef mpiNumberType
  #undef mpiNumberIntType

  #define mpiNumberType ((medi::MpiTypeInterface*)TOOL->MPI_TYPE)
  #define mpiNumberIntType ((medi::MpiTypeInterface*)TOOL->MPI_INT_TYPE)
#endif
