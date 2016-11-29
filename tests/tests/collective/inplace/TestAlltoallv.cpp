#include <toolDefines.h>

IN(10)
OUT(10)
POINTS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};
SEEDS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0}}};

void func(NUMBER* x, NUMBER* y) {
  int world_rank;
  medi::AMPI_Comm_rank(AMPI_COMM_WORLD, &world_rank);
  int world_size;
  medi::AMPI_Comm_size(AMPI_COMM_WORLD, &world_size);

  for(int i = 0; i < 10; ++i) {
    y[i] = x[i];
  }
  int counts[2] = {5, 5};
  int displs[2] = {0, 5};
  medi::AMPI_Alltoallv(medi::AMPI_IN_PLACE, NULL, NULL, mpiNumberType, y, counts, displs, mpiNumberType, MPI_COMM_WORLD);
}
