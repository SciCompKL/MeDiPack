#pragma once

#include <adolc/adolc.h>
#include <medi/medi.hpp>
#include <medi/adolcMeDiPackTypes.hpp>

typedef adouble NUMBER;

#include "../globalDefines.h"

#define TOOL AdolcTool
typedef medi::MpiTypeDefault<TOOL> MPI_NUMBER;
extern MPI_NUMBER* mpiNumberType;
