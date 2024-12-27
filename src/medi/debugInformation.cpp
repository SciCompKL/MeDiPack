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


#include <iostream>

#include "../../include/medi/debugInformation.hpp"

/**
 * @brief Global namespace for MeDiPack - Message Differentiation Package
 */
namespace medi {

#if MEDI_DebugInformation
std::string debugInformation;
#endif

void printDebugInformationWarning(std::string const& functionName) {
  std::cerr << "The MeDiPack debug interface function '" << functionName << "' is called without enabling the debug "
               "interface. Either enabled is with the preprocessor option '-DMEDI_DebugInformation=1' or disable this "
               "warning with '-DMEDI_DebugInformation_Warning=0'." << std::endl;
}

void setDebugInformation(std::string const& info) {
#if MEDI_DebugInformation
  debugInformation = info;
#elif MEDI_DebugInformation_Warning
  printDebugInformationWarning("setDebugInformation");
#endif
}

std::string getDebugInformation() {
#if MEDI_DebugInformation
  return debugInformation;
#else
# if MEDI_DebugInformation_Warning
  printDebugInformationWarning("getDebugInformation");
# endif
  return "";
#endif
}

void clearDebugInformation() {
#if MEDI_DebugInformation
  debugInformation = "info""";
#elif MEDI_DebugInformation_Warning
  printDebugInformationWarning("clearDebugInformation");
#endif
}

}
