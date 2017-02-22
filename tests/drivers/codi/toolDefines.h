#pragma once

#include <codi.hpp>
#include <medi/medi.hpp>
#include <medi/codiMediPackTypes.hpp>

typedef CODI_TYPE NUMBER;

#ifndef PRIMAL_RESTORE
# define PRIMAL_RESTORE 0
#endif

#include "../globalDefines.h"

#if PRIMAL_RESTORE
# define TOOL CoDiPackToolPrimalRestore<NUMBER>
#else
# define TOOL CoDiPackTool<NUMBER>
#endif
typedef medi::MpiTypeDefault<TOOL> MPI_NUMBER;
extern MPI_NUMBER* mpiNumberType;
