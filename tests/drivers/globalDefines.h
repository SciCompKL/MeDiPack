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

#define POINTS(number) \
  int getEvalPointsCount() {return number;} \
  extern double points[number][2][in_count]; \
  double getEvalPoint(int point, int rank, int col) { return points[point][rank][col]; } \
  double points[number][2][in_count]

#define SEEDS(number) \
  int getEvalSeedCount() {return number;} \
  extern double seeds[number][2][out_count]; \
  double getEvalSeed(int point, int rank, int col) { return seeds[point][rank][col]; } \
  double seeds[number][2][out_count]

#define IN(number) \
  const int in_count = number; \
  int getInputCount() {return number;}

#define OUT(number) \
  const int out_count = number; \
  int getOutputCount() {return number;}

int getEvalPointsCount();
double getEvalPoint(int point, int rank, int col);
double getEvalSeed(int point, int rank, int col);
int getInputCount();
int getOutputCount();

void func(NUMBER* x, NUMBER* y);

#define mpiNumberType TOOL->MPI_TYPE
#define mpiNumberIntType TOOL->MPI_INT_TYPE
