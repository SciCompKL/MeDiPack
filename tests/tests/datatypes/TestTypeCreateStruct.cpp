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
#include <cstddef>

IN(10)
OUT(10)
POINTS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};
SEEDS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};

struct TestStruct {
  NUMBER d[4];
  int i;
  NUMBER f;
};

void func(NUMBER* x, NUMBER* y) {
  int world_rank;
  medi::AMPI_Comm_rank(AMPI_COMM_WORLD, &world_rank);
  int world_size;
  medi::AMPI_Comm_size(AMPI_COMM_WORLD, &world_size);

  int blockLength[3] = {4, 1, 1};
  AMPI_Aint offsets[3] = {
    offsetof(TestStruct, d),
    offsetof(TestStruct, i),
    offsetof(TestStruct, f)
  };
  const medi::AMPI_Datatype types[3] = {mpiNumberType, medi::AMPI_INT, mpiNumberType};

  medi::AMPI_Datatype testType;
  medi::AMPI_Type_create_struct(3, blockLength, offsets, types, &testType);
  medi::AMPI_Type_commit(&testType);

  TestStruct data[2];

  if(world_rank == 0) {
    size_t offset = 0;
    for(int i = 0; i < 2; ++i) {
      data[i].d[0] = x[offset++];
      data[i].d[1] = x[offset++];
      data[i].d[2] = x[offset++];
      data[i].d[3] = x[offset++];
      data[i].f = x[offset++];

      data[i].i = i + 1;
    }
    medi::AMPI_Send(data, 2, testType, 1, 42, AMPI_COMM_WORLD);
  } else {
    medi::AMPI_Recv(data, 2, testType, 0, 42, AMPI_COMM_WORLD, AMPI_STATUS_IGNORE);

    size_t offset = 0;
    for(int i = 0; i < 2; ++i) {
      y[offset++] = data[i].d[0];
      y[offset++] = data[i].d[1];
      y[offset++] = data[i].d[2];
      y[offset++] = data[i].d[3];
      y[offset++] = data[i].f;

      mediAssert(i + 1 == data[i].i);
    }
  }

  // We do not free the type here since it is required for the reverse evaluation
  //medi::AMPI_Type_free(&testType);
}
