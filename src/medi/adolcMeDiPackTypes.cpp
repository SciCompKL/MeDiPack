#include "../../include/medi/adolcMeDiPackTypes.hpp"

MPI_Datatype AdolcTool::MpiType;
MPI_Datatype AdolcTool::ModifiedMpiType;
MPI_Datatype AdolcTool::AdjointMpiType;
medi::AMPI_Op AdolcTool::OP_SUM;
medi::AMPI_Op AdolcTool::OP_PROD;
medi::AMPI_Op AdolcTool::OP_MIN;
medi::AMPI_Op AdolcTool::OP_MAX;

double* AdolcTool::adjointBase;
double* AdolcTool::primalBase;
ext_diff_fct_v2* AdolcTool::extFunc;

bool AdolcTool::deleteReverseHandles = true;
