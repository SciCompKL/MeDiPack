/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2015-2025 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://scicomp.rptu.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of MeDiPack (http://scicomp.rptu.de/software/medi).
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

#include <mpi.h>

#define MEDI_MAJOR_VERSION 1
#define MEDI_MINOR_VERSION 3
#define MEDI_BUILD_VERSION 3
#define MEDI_VERSION "1.3.3"


namespace medi {
#ifndef MEDI_HeaderOnly
  /// See medi::HeaderOnly.
  #define MEDI_HeaderOnly true
#endif
    /// If MeDiPack is included as header only library or as a regular library.
    bool constexpr HeaderOnly = MEDI_HeaderOnly;
}

#include "ampi/ampi.hpp"
