/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2017 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, TU Kaiserslautern)
 *
 * This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * MeDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * MeDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with MeDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum (SciComp, TU Kaiserslautern)
 */

#pragma once

#include <mpi.h>

#include "ampiDefinitions.h"

namespace medi {

  AMPI_Op AMPI_MAX;
  AMPI_Op AMPI_MIN;
  AMPI_Op AMPI_SUM;
  AMPI_Op AMPI_PROD;
  AMPI_Op AMPI_MAXLOC;
  AMPI_Op AMPI_MINLOC;
  AMPI_Op AMPI_BAND;
  AMPI_Op AMPI_BOR;
  AMPI_Op AMPI_BXOR;
  AMPI_Op AMPI_LAND;
  AMPI_Op AMPI_LOR;
  AMPI_Op AMPI_LXOR;
  AMPI_Op AMPI_REPLACE;
  AMPI_Op AMPI_NO_OP;

  AMPI_CHAR_Type* AMPI_CHAR;
  AMPI_SHORT_Type* AMPI_SHORT;
  AMPI_INT_Type* AMPI_INT;
  AMPI_LONG_Type* AMPI_LONG;
  AMPI_LONG_LONG_INT_Type* AMPI_LONG_LONG_INT;
  AMPI_LONG_LONG_Type* AMPI_LONG_LONG;
  AMPI_SIGNED_CHAR_Type* AMPI_SIGNED_CHAR;
  AMPI_UNSIGNED_CHAR_Type* AMPI_UNSIGNED_CHAR;
  AMPI_UNSIGNED_SHORT_Type* AMPI_UNSIGNED_SHORT;
  AMPI_UNSIGNED_Type* AMPI_UNSIGNED;
  AMPI_UNSIGNED_LONG_Type* AMPI_UNSIGNED_LONG;
  AMPI_UNSIGNED_LONG_LONG_Type* AMPI_UNSIGNED_LONG_LONG;
  AMPI_FLOAT_Type* AMPI_FLOAT;
  AMPI_DOUBLE_Type* AMPI_DOUBLE;
  AMPI_LONG_DOUBLE_Type* AMPI_LONG_DOUBLE;
  AMPI_WCHAR_Type* AMPI_WCHAR;
  AMPI_C_BOOL_Type* AMPI_C_BOOL;
  AMPI_INT8_T_Type* AMPI_INT8_T;
  AMPI_INT16_T_Type* AMPI_INT16_T;
  AMPI_INT32_T_Type* AMPI_INT32_T;
  AMPI_INT64_T_Type* AMPI_INT64_T;
  AMPI_UINT8_T_Type* AMPI_UINT8_T;
  AMPI_UINT16_T_Type* AMPI_UINT16_T;
  AMPI_UINT32_T_Type* AMPI_UINT32_T;
  AMPI_UINT64_T_Type* AMPI_UINT64_T;
  AMPI_AINT_Type* AMPI_AINT;
  AMPI_COUNT_Type* AMPI_COUNT;
  AMPI_OFFSET_Type* AMPI_OFFSET;
  AMPI_BYTE_Type* AMPI_BYTE;
  AMPI_PACKED_Type* AMPI_PACKED;
  AMPI_CXX_BOOL_Type* AMPI_CXX_BOOL;
  AMPI_FLOAT_INT_Type* AMPI_FLOAT_INT;
  AMPI_DOUBLE_INT_Type* AMPI_DOUBLE_INT;
  AMPI_LONG_INT_Type* AMPI_LONG_INT;
  AMPI_2INT_Type* AMPI_2INT;
  AMPI_SHORT_INT_Type* AMPI_SHORT_INT;
  AMPI_LONG_DOUBLE_INT_Type* AMPI_LONG_DOUBLE_INT;

  void initializeOperators() {
    AMPI_MAX.init(MPI_MAX);
    AMPI_MIN.init(MPI_MIN);
    AMPI_SUM.init(MPI_SUM);
    AMPI_PROD.init(MPI_PROD);
    AMPI_MAXLOC.init(MPI_MAXLOC);
    AMPI_MINLOC.init(MPI_MINLOC);
    AMPI_BAND.init(MPI_BAND);
    AMPI_BOR.init(MPI_BOR);
    AMPI_BXOR.init(MPI_BXOR);
    AMPI_LAND.init(MPI_LAND);
    AMPI_LOR.init(MPI_LOR);
    AMPI_LXOR.init(MPI_LXOR);
    AMPI_REPLACE.init(MPI_REPLACE);
    AMPI_NO_OP.init(MPI_NO_OP);
  }

  void initTypes() {
    AMPI_CHAR = new AMPI_CHAR_Type(MPI_CHAR);
    AMPI_SHORT = new AMPI_SHORT_Type(MPI_SHORT);
    AMPI_INT = new AMPI_INT_Type(MPI_INT);
    AMPI_LONG = new AMPI_LONG_Type(MPI_LONG);
    AMPI_LONG_LONG_INT = new AMPI_LONG_LONG_INT_Type(MPI_LONG_LONG_INT);
    AMPI_LONG_LONG = new AMPI_LONG_LONG_Type(MPI_LONG_LONG);
    AMPI_SIGNED_CHAR = new AMPI_SIGNED_CHAR_Type(MPI_SIGNED_CHAR);
    AMPI_UNSIGNED_CHAR = new AMPI_UNSIGNED_CHAR_Type(MPI_UNSIGNED_CHAR);
    AMPI_UNSIGNED_SHORT = new AMPI_UNSIGNED_SHORT_Type(MPI_UNSIGNED_SHORT);
    AMPI_UNSIGNED = new AMPI_UNSIGNED_Type(MPI_UNSIGNED);
    AMPI_UNSIGNED_LONG = new AMPI_UNSIGNED_LONG_Type(MPI_UNSIGNED_LONG);
    AMPI_UNSIGNED_LONG_LONG = new AMPI_UNSIGNED_LONG_LONG_Type(MPI_UNSIGNED_LONG_LONG);
    AMPI_FLOAT = new AMPI_FLOAT_Type(MPI_FLOAT);
    AMPI_DOUBLE = new AMPI_DOUBLE_Type(MPI_DOUBLE);
    AMPI_LONG_DOUBLE = new AMPI_LONG_DOUBLE_Type(MPI_LONG_DOUBLE);
    AMPI_WCHAR = new AMPI_WCHAR_Type(MPI_WCHAR);
    AMPI_C_BOOL = new AMPI_C_BOOL_Type(MPI_C_BOOL);
    AMPI_INT8_T = new AMPI_INT8_T_Type(MPI_INT8_T);
    AMPI_INT16_T = new AMPI_INT16_T_Type(MPI_INT16_T);
    AMPI_INT32_T = new AMPI_INT32_T_Type(MPI_INT32_T);
    AMPI_INT64_T = new AMPI_INT64_T_Type(MPI_INT64_T);
    AMPI_UINT8_T = new AMPI_UINT8_T_Type(MPI_UINT8_T);
    AMPI_UINT16_T = new AMPI_UINT16_T_Type(MPI_UINT16_T);
    AMPI_UINT32_T = new AMPI_UINT32_T_Type(MPI_UINT32_T);
    AMPI_UINT64_T = new AMPI_UINT64_T_Type(MPI_UINT64_T);
    AMPI_AINT = new AMPI_AINT_Type(MPI_AINT);
    AMPI_COUNT = new AMPI_COUNT_Type(MPI_COUNT);
    AMPI_OFFSET = new AMPI_OFFSET_Type(MPI_OFFSET);
    AMPI_BYTE = new AMPI_BYTE_Type(MPI_BYTE);
    AMPI_PACKED = new AMPI_PACKED_Type(MPI_PACKED);
    AMPI_CXX_BOOL = new AMPI_CXX_BOOL_Type(MPI_CXX_BOOL);
    AMPI_FLOAT_INT = new AMPI_FLOAT_INT_Type(MPI_FLOAT_INT);
    AMPI_DOUBLE_INT = new AMPI_DOUBLE_INT_Type(MPI_DOUBLE_INT);
    AMPI_LONG_INT = new AMPI_LONG_INT_Type(MPI_LONG_INT);
    AMPI_2INT = new AMPI_2INT_Type(MPI_2INT);
    AMPI_SHORT_INT = new AMPI_SHORT_INT_Type(MPI_SHORT_INT);
    AMPI_LONG_DOUBLE_INT = new AMPI_LONG_DOUBLE_INT_Type(MPI_LONG_DOUBLE_INT);
  }

  void finalizeTypes() {
    delete AMPI_CHAR;
    delete AMPI_SHORT;
    delete AMPI_INT;
    delete AMPI_LONG;
    delete AMPI_LONG_LONG_INT;
    delete AMPI_LONG_LONG;
    delete AMPI_SIGNED_CHAR;
    delete AMPI_UNSIGNED_CHAR;
    delete AMPI_UNSIGNED_SHORT;
    delete AMPI_UNSIGNED;
    delete AMPI_UNSIGNED_LONG;
    delete AMPI_UNSIGNED_LONG_LONG;
    delete AMPI_FLOAT;
    delete AMPI_DOUBLE;
    delete AMPI_LONG_DOUBLE;
    delete AMPI_WCHAR;
    delete AMPI_C_BOOL;
    delete AMPI_INT8_T;
    delete AMPI_INT16_T;
    delete AMPI_INT32_T;
    delete AMPI_INT64_T;
    delete AMPI_UINT8_T;
    delete AMPI_UINT16_T;
    delete AMPI_UINT32_T;
    delete AMPI_UINT64_T;
    delete AMPI_AINT;
    delete AMPI_COUNT;
    delete AMPI_OFFSET;
    delete AMPI_BYTE;
    delete AMPI_PACKED;
    delete AMPI_CXX_BOOL;
    delete AMPI_FLOAT_INT;
    delete AMPI_DOUBLE_INT;
    delete AMPI_LONG_INT;
    delete AMPI_2INT;
    delete AMPI_SHORT_INT;
    delete AMPI_LONG_DOUBLE_INT;
  }
}
