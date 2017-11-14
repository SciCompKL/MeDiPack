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

#include <toolDefines.h>

#include <iostream>
#include <vector>

int main(int nargs, char** args) {

  medi::AMPI_Init(&nargs, &args);

  int world_rank;
  medi::AMPI_Comm_rank(AMPI_COMM_WORLD, &world_rank);
  int world_size;
  medi::AMPI_Comm_size(AMPI_COMM_WORLD, &world_size);


  TOOL::init();

  int evalPoints = getEvalPointsCount();
  int inputs = getInputCount();
  int outputs = getOutputCount();
  NUMBER* x = new NUMBER[inputs];
  NUMBER* y = new NUMBER[outputs];

  for(int curPoint = 0; curPoint < evalPoints; ++curPoint) {
    std::cout << "Point " << curPoint << " : {";

    for(int i = 0; i < inputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalPoint(curPoint, world_rank, i);
      std::cout << val;

      x[i] = (NUMBER)(val);
    }
    std::cout << "}\n";

    for(int i = 0; i < outputs; ++i) {
      y[i] = 0.0;
    }

    std::cout << "Seed " << curPoint << " : {";
    for(int i = 0; i < inputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalSeed(curPoint, world_rank, i);
      std::cout << val;
#if VECTOR
      x[i].gradient()[0] = val;
#else
      x[i].gradient() = val;
#endif
    }
    std::cout << "}\n";

    func(x, y);

    for(int curIn = 0; curIn < inputs; ++curIn) {
      double grad;
#if VECTOR
      grad = y[curIn].gradient()[0];
#else
      grad = y[curIn].gradient();
#endif
      std::cout << curIn << " " << grad << std::endl;
    }
  }

  delete [] y;
  delete [] x;

  TOOL::finalize();
  medi::AMPI_Finalize();
}

#include <medi/medi.cpp>
