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

#include <toolDefines.h>

#if CODI_MAJOR_VERSION >= 3
template<typename Tape>
struct Tester {
    using Real = typename Tape::Real;
    using Identifier = typename Tape::Identifier;

    using EvalHandle = typename Tape::EvalHandle;

    int lastCount = 0;
    int llfCount = 0;

    Tape& tape;

    Tester(Tape& tape) : tape(tape) {}

    void count(Identifier id) {
      lastCount += 1;
    }

    void handleStatement(Identifier& lhsIndex, codi::Config::ArgumentSize const& size, Real const* jacobians,
                         Identifier const* rhsIdentifiers) {
      // Ignore statements
    }

    void handleStatement(EvalHandle const& evalHandle, codi::Config::ArgumentSize const& nPassiveValues, size_t& linearAdjointPosition, char* stmtData) {
      // Ignore statements
    }

    void handleLowLevelFunction(codi::LowLevelFunctionEntry<Tape, Real, Identifier> const& func, codi::ByteDataView& llfData) {
      func.template call<codi::LowLevelFunctionEntryCallKind::IterateOutputs>(&tape, llfData, reinterpret_cast<void (*)(int*, void*)>(countId), this);
      int outputs = lastCount;
      lastCount = 0;

      llfData.reset();
      func.template call<codi::LowLevelFunctionEntryCallKind::IterateInputs>(&tape, llfData, reinterpret_cast<void (*)(int*, void*)>(countId), this);
      int inputs = lastCount;
      lastCount = 0;

      if(llfCount == 0) {
        if (!(outputs == 0 && inputs == 10)) {
          std::cout << "Error: Input/output count is wrong expected 10/0 got " << inputs << "/" << outputs << std::endl;
        }
      } else if(llfCount == 1) {
        if (!(outputs == 10 && inputs == 0)) {
          std::cout << "Error: Input/output count is wrong expected 0/10 got " << inputs << "/" << outputs << std::endl;
        }
      } else {
        std::cout << "To many low level functions expected only 2." << std::endl;
      }

      llfCount += 1;
    }

    static void countId(Identifier* id, Tester* data) {
      data->count(*id);
    }
};
#endif

IN(10)
OUT(10)
POINTS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};
SEEDS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};

void func(NUMBER* x, NUMBER* y) {
  medi::AMPI_Request request;
  medi::AMPI_Iallreduce(x, &y[ 0], 10, mpiNumberType, medi::AMPI_SUM, MPI_COMM_WORLD, &request);

  medi::AMPI_Wait(&request, AMPI_STATUS_IGNORE);

#if CODI_MAJOR_VERSION >= 3
  Tester<typename NUMBER::Tape> tester = {NUMBER::getTape()};
  NUMBER::getTape().iterateForward(tester);
#endif
}
