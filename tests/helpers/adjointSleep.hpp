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
#include <unistd.h>

#ifdef CODI_MAJOR_VERSION
template<typename Tape>
void sleep_rev(Tape* t, void* d, codi::VectorAccessInterface<typename Tape::Real, typename Tape::Identifier>* va) {
  int* data = (int*)d;

  usleep(*data);
}

template<typename Tape>
void sleep_del(Tape* t, void* d) {
  int* data = (int*)d;

  delete data;
}

template<typename Real>
void addAdjointSleep(int microseconds) {
  using Tape = typename Real::Tape;
  Tape& tape = Real::getTape();

  int* data = new int(microseconds);
  tape.pushExternalFunction(codi::ExternalFunction<Tape>::create(sleep_rev<Tape>, data, sleep_del<Tape>));
}
#else
template<typename Real>
void addAdjointSleep(int microseconds) {
  std::cerr << "Adjoint sleep not implemented for tool." << std::endl;
}

#endif
