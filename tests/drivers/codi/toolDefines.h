/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2017-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
 * Authors: Max Sagebaum, Tim Albring (SciComp, TU Kaiserslautern)
 */

#pragma once

#include <codi.hpp>
#include <medi/medi.hpp>
#include <codi/externals/codiMpiTypes.hpp>

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

#define TOOL_TYPE CoDiMpiTypes<NUMBER>
#define TOOL codiTypes

extern TOOL_TYPE* codiTypes;

#include "../globalDefines.h"

#if UNTYPED
  #undef mpiNumberType
  #undef mpiNumberIntType

  #define mpiNumberType ((medi::MpiTypeInterface*)TOOL->MPI_TYPE)
  #define mpiNumberIntType ((medi::MpiTypeInterface*)TOOL->MPI_INT_TYPE)
#endif
