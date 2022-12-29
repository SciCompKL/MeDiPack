/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
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

#include <toolDefines.h>

#include <adolc/interfaces.h>
#include <adolc/taping.h>
#include <adolc/drivers/drivers.h>

#include <iostream>
#include <vector>

int main(int nargs, char** args) {

  medi::AMPI_Init(&nargs, &args);

  int world_rank;
  medi::AMPI_Comm_rank(AMPI_COMM_WORLD, &world_rank);
  int world_size;
  medi::AMPI_Comm_size(AMPI_COMM_WORLD, &world_size);

  TOOL = new TOOL_TYPE();

  int evalPoints = getEvalPointsCount();
  int inputs = getInputCount();
  int outputs = getOutputCount();
  NUMBER* x = new NUMBER[inputs];
  NUMBER* y = new NUMBER[outputs];

  double* grad = new double[inputs];
  double* seed = new double[outputs];

  for(int curPoint = 0; curPoint < evalPoints; ++curPoint) {

    trace_on(1, 1);
    std::cout << "Point " << curPoint << " : {";

    for(int i = 0; i < inputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalPoint(curPoint, world_rank, i);
      std::cout << val;

      x[i] <<= val;
    }
    std::cout << "}\n";

    for(int i = 0; i < outputs; ++i) {
      y[i] = 0.0;
    }

    func(x, y);

    for(int i = 0; i < outputs; ++i) {
      double temp;
      y[i] >>= temp;
    }

    trace_off();

    std::cout << "Seed " << curPoint << " : {";
    for(int i = 0; i < outputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalSeed(curPoint, world_rank, i);
      std::cout << val;

      seed[i] = val;
    }
    std::cout << "}\n";

    fos_reverse((short)1, (int)outputs, (int)inputs, seed, grad);

    for(int curIn = 0; curIn < inputs; ++curIn) {
      std::cout << curIn << " " << grad[curIn] << std::endl;
    }


  }

  delete [] grad;
  delete [] seed;
  delete [] y;
  delete [] x;

  delete TOOL;

  medi::AMPI_Finalize();
}

TOOL_TYPE* TOOL;

#include <medi/medi.cpp>
