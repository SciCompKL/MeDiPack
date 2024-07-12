/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://scicomp.rptu.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of MeDiPack (http://scicomp.rptu.de/software/codi).
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

#include <iostream>
#include <vector>

#if CODI_MAJOR_VERSION >= 2
  using Tape = NUMBER::Tape;
  using Real = NUMBER::Real;

  Tape& getTape() { return NUMBER::getTape(); }
#if PRIMAL_TAPE
  Real& primalValueFromTape(Tape& tape, NUMBER& value) { return tape.primal(value.getIdentifier()); }
#endif
#else
  using Tape = NUMBER::TapeType;
  using Real = NUMBER::Real;

  Tape& getTape() { return NUMBER::getGlobalTape(); }
#if PRIMAL_TAPE
  Real& primalValueFromTape(Tape& tape, NUMBER& value) { return tape.primalValue(value.getGradientData()); }
#endif
#endif

void seedValues(size_t curPoint, size_t world_rank, NUMBER* vec, size_t size) {
  Tape& tape = getTape();
  (void)tape;

  for(size_t i = 0; i < size; ++i) {
    if(i != 0) {
      std::cout << ", ";
    }
    double val = getEvalSeed(curPoint, world_rank, i);
    std::cout << val;
#if PRIMAL_TAPE
    primalValueFromTape(tape, vec[i]) = val;
#elif VECTOR
    vec[i].gradient()[0] = val;
#else
    vec[i].gradient() = val;
#endif
  }
}

void outputGradient(NUMBER* vec, size_t size) {
  Tape& tape = getTape();
  (void)tape;

  for(size_t i = 0; i < size; ++i) {
    NUMBER::Real grad;
#if PRIMAL_TAPE
    grad = primalValueFromTape(tape, vec[i]);
#elif VECTOR
    grad = vec[i].gradient()[0];
#else
    grad = vec[i].gradient();
#endif
    std::cout << i << " " << grad << std::endl;
  }
}

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

  Tape& tape = getTape();
  tape.setActive();

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

    for(int i = 0; i < inputs; ++i) {
      tape.registerInput(x[i]);
    }

    func(x, y);

    for(int i = 0; i < outputs; ++i) {
      tape.registerOutput(y[i]);
    }

    std::cout << "Seed " << curPoint << " : {";

#if FORWARD_TAPE
    seedValues(curPoint, world_rank, x, inputs);
#elif PRIMAL_TAPE
    seedValues(curPoint, world_rank, x, inputs);
#else
    seedValues(curPoint, world_rank, y, outputs);
#endif

    std::cout << "}\n";


#if FORWARD_TAPE
    tape.evaluateForward();
    outputGradient(y, outputs);
#elif PRIMAL_TAPE
    tape.evaluatePrimal();
    outputGradient(y, outputs);
#else
    tape.evaluate();
    outputGradient(x, inputs);
#endif

    tape.reset();
  }

  delete [] y;
  delete [] x;

  delete TOOL;

  medi::AMPI_Finalize();
}

TOOL_TYPE* TOOL;

#include <medi/medi.cpp>
