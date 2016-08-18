#pragma once

#include <codi.hpp>
#include <medi/medi.hpp>
#include <medi/codiMediPackTypes.hpp>

typedef codi::RealReverse NUMBER;

#include "../globalDefines.h"

#define TOOL CoDiPackTool<NUMBER>
typedef medi::MpiTypeInterface MPI_NUMBER;
extern MPI_NUMBER* mpiNumberType;
