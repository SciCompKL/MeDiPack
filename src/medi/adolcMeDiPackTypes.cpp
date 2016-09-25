#include "../../include/medi/adolcMeDiPackTypes.hpp"

MPI_Datatype AdolcTool::MpiType;
MPI_Datatype AdolcTool::ModifiedMpiType;
MPI_Datatype AdolcTool::AdjointMpiType;
medi::AMPI_Op AdolcTool::OP_ADD;
medi::AMPI_Op AdolcTool::OP_MUL;
medi::AMPI_Op AdolcTool::OP_MIN;
medi::AMPI_Op AdolcTool::OP_MAX;

double* AdolcTool::adjointBase;
double* AdolcTool::primalBase;
