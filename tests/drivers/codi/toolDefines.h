#pragma once

#include <codi.hpp>
#include <medi/medi.hpp>
#include <medi/codiMediPackTypes.hpp>

typedef CODI_TYPE NUMBER;

#include "../globalDefines.h"

#define TOOL CoDiPackTool<NUMBER>
typedef medi::MpiTypeDefault<TOOL> MPI_NUMBER;
extern MPI_NUMBER* mpiNumberType;
