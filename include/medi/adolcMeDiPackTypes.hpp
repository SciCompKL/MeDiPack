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

#include <adolc/adolc.h>
#include <adolc/externfcts2.h>

#include "medipack.h"
#include "ampi/async.hpp"
#include "exceptions.hpp"

#include "adToolInterface.h"

template <typename T>
void adolcModifiedAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] += invec[i];
  }
}

template <typename T>
void adolcModifiedMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] *= invec[i];
  }
}

template <typename T>
void adolcModifiedMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = max(inoutvec[i], invec[i]);
  }
}

template <typename T>
void adolcModifiedMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = min(inoutvec[i], invec[i]);
  }
}

template <typename T>
void adolcUnmodifiedAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] += invec[i];
  }
}

template <typename T>
void adolcUnmodifiedMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] *= invec[i];
  }
}

template <typename T>
void adolcUnmodifiedMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = max(inoutvec[i], invec[i]);
  }
}

template <typename T>
void adolcUnmodifiedMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = min(inoutvec[i], invec[i]);
  }
}

template <typename AT, typename PT>
void adolcPreAdjMul(AT* adjoints, PT* primals, int count) {
  for(int i = 0; i < count; ++i) {
    adjoints[i] *= primals[i];
  }
}

template <typename AT, typename PT>
void adolcPostAdjMul(AT* adjoints, PT* primals, PT* rootPrimals, int count) {
  CODI_UNUSED(rootPrimals);

  for(int i = 0; i < count; ++i) {
    if(0.0 != primals[i]) {
      adjoints[i] /= primals[i];
    }
  }
}

template <typename AT, typename PT>
void adolcPostAdjMinMax(AT* adjoints, PT* primals, PT* rootPrimals, int count) {
  for(int i = 0; i < count; ++i) {
    if(rootPrimals[i] != primals[i]) {
      adjoints[i] = 0.0; // the primal of this process was not the minimum or maximum so do not perfrom the adjoint update
    }
  }
}

struct AdolcTool final : public medi::ADToolBase<AdolcTool, double, double, int> {
  typedef adouble Type;
  typedef double AdjointType;
  typedef double ModifiedType;
  typedef double PassiveType;
  typedef int IndexType;

  const static bool IS_ActiveType = true;
  const static bool IS_RequiresModifiedBuffer = true;

  static MPI_Datatype MpiType;
  static MPI_Datatype ModifiedMpiType;
  static MPI_Datatype AdjointMpiType;
  static medi::AMPI_Op OP_SUM;
  static medi::AMPI_Op OP_PROD;
  static medi::AMPI_Op OP_MIN;
  static medi::AMPI_Op OP_MAX;

  static double* adjointBase;
  static double* primalBase;
  static ext_diff_fct_v2 *extFunc;

  static bool deleteReverseHandles;

  static void setDeleteReverseHandles(bool value) {
    deleteReverseHandles = value;
  }

  static bool isDeleteReverseHandles() {
    return deleteReverseHandles;
  }

  static void initTypes() {
    // create the mpi type for ADOL-c
    MpiType = MPI_INT;
    ModifiedMpiType = MPI_DOUBLE;
    AdjointMpiType = MPI_DOUBLE;
  }

  static void initOperator(medi::AMPI_Op& op, bool requiresPrimal, bool requiresPrimalSend, MPI_User_function* modifiedFunc, MPI_User_function* primalFunc, const medi::PreAdjointOperation preAdjointOperation, const medi::PostAdjointOperation postAdjointOperation) {
    op.init(requiresPrimal, requiresPrimalSend, primalFunc, true, modifiedFunc, true, preAdjointOperation, postAdjointOperation);
  }

  static void initOperator(medi::AMPI_Op& op, MPI_User_function* primalFunc) {
    op.init(primalFunc, true);
  }

  static void initOperators() {
    initOperator(OP_SUM, false, false, (MPI_User_function*)adolcModifiedAdd<ModifiedType>, (MPI_User_function*)adolcUnmodifiedAdd<Type>, medi::noPreAdjointOperation, medi::noPostAdjointOperation);
    initOperator(OP_PROD, (MPI_User_function*)adolcUnmodifiedMul<Type>);
    initOperator(OP_MIN, true, true, (MPI_User_function*)adolcModifiedMin<ModifiedType>, (MPI_User_function*)adolcUnmodifiedMin<Type>, medi::noPreAdjointOperation, (medi::PostAdjointOperation)adolcPostAdjMinMax<AdjointType, PassiveType>);
    initOperator(OP_MAX, true, true, (MPI_User_function*)adolcModifiedMax<ModifiedType>, (MPI_User_function*)adolcUnmodifiedMax<Type>, medi::noPreAdjointOperation, (medi::PostAdjointOperation)adolcPostAdjMinMax<AdjointType, PassiveType>);
  }

  static void initExternalFunction() {
    extFunc = reg_ext_fct(emptyPrimal);
    extFunc->fos_reverse = callHandle;
  }

  static void init() {
    initTypes();
    initOperators();
    initExternalFunction();
  }

  static void finalizeOperators() {
    OP_SUM.free();
    OP_PROD.free();
    OP_MIN.free();
    OP_MAX.free();
  }

  static void finalizeTypes() {
  }

  static void finalize() {
    finalizeOperators();
    finalizeTypes();
  }

  AdolcTool(MPI_Datatype adjointMpiType) :
    medi::ADToolBase<AdolcTool, double, double, int>(adjointMpiType) {}


  inline bool isActiveType() const {
    return true;
  }

  inline  bool isHandleRequired() const {
    return isTaping();
  }

  inline bool isOldPrimalsRequired() const {
    return true;
  }

  inline void startAssembly(medi::HandleBase* h) {
    MEDI_UNUSED(h);

  }

  inline void addToolAction(medi::HandleBase* h) {
    if(NULL != h) {
      // store the handle pointer in the int array
      int sizePointer = sizeof(medi::HandleBase*) / sizeof(int);
      if(sizePointer * sizeof(int) != sizeof(medi::HandleBase*)) {
        MEDI_EXCEPTION("Error pointer type is not a multiple of an int.");
      }

      call_ext_fct(extFunc, sizePointer, reinterpret_cast<int*>(&h), 0, 0, NULL, NULL, NULL, NULL);
    }
  }

  inline void stopAssembly(medi::HandleBase* h) {
    MEDI_UNUSED(h);
  }

  inline void getAdjoints(const IndexType* indices, AdjointType* adjoints, int elements) const {
    for(int pos = 0; pos < elements; ++pos) {
      adjoints[pos] = adjointBase[indices[pos]];
      adjointBase[indices[pos]] = 0.0;
    }
  }

  inline void updateAdjoints(const IndexType* indices, const AdjointType* adjoints, int elements) const {
    for(int pos = 0; pos < elements; ++pos) {
      adjointBase[indices[pos]] += adjoints[pos];
    }
  }

  inline void setReverseValues(const IndexType* indices, const PassiveType* primals, int elements) const {
    for(int pos = 0; pos < elements; ++pos) {
      primalBase[indices[pos]] = primals[pos];
    }
  }

  inline void combineAdjoints(AdjointType* buf, const int elements, const int ranks) const {
    for(int curRank = 1; curRank < ranks; ++curRank) {
      for(int curPos = 0; curPos < elements; ++curPos) {
        buf[curPos] += buf[elements * curRank + curPos];
      }
    }
  }

  inline void createAdjointTypeBuffer(AdjointType* &buf, size_t size) const {
    buf = new AdjointType[size];
  }

  inline void createPassiveTypeBuffer(PassiveType* &buf, size_t size) const {
    buf = new PassiveType[size];
  }

  inline void createIndexTypeBuffer(IndexType* &buf, size_t size) const {
    buf = new IndexType[size];
  }

  inline void deleteAdjointTypeBuffer(AdjointType* &buf) const {
    if(NULL != buf) {
      delete [] buf;
      buf = NULL;
    }
  }

  inline void deletePassiveTypeBuffer(PassiveType* &buf) const {
    if(NULL != buf) {
      delete [] buf;
      buf = NULL;
    }
  }

  inline void deleteIndexTypeBuffer(IndexType* &buf) const {
    if(NULL != buf) {
      delete [] buf;
      buf = NULL;
    }
  }

  static int emptyPrimal(int iArrLen, int *iArr, int nin, int nout, int *insz, double **x, int *outsz, double **y, void* ctx) {
    return 0;
  }

  static int callHandle(int iArrLen, int* iArr, int nout, int nin, int *outsz, double **up, int *insz, double **zp, double **x, double **y, void *h) {

    medi::HandleBase* handle = *reinterpret_cast<medi::HandleBase**>(iArr);
    adjointBase = extFunc->adjointVector;
    primalBase = extFunc->primalVector;
    handle->func(handle);

    if(deleteReverseHandles) {
      switch (handle->deleteType) {
      case medi::ManualDeleteType::Normal:
        delete handle;
        break;
      case medi::ManualDeleteType::Async:
        // handle is part of an asyncronous communication
        // it will be deleted in the correspoinding wait
        break;
      case medi::ManualDeleteType::Wait:
        // delete the wait handle and the corresponding asyncrounous handle
        {
          medi::WaitHandle* waitHandle = static_cast<medi::WaitHandle*>(handle);
          delete waitHandle->handle;
          delete waitHandle;
        }
        break;
      default:
        MEDI_EXCEPTION("Not implemented switch case.");
        break;
      }

    }

    return 0;
  }

  static inline int getIndex(const Type& value) {
    return value.loc();
  }

  static inline void clearIndex(Type& value) {
    // do nothing
  }

  static inline PassiveType getValue(const Type& value) {
    return value.value();
  }

  static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) {
    modValue = value.value();
  }

  static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) {
    value.setValue(modValue);
  }

  static inline int registerValue(Type& value, PassiveType& oldPrimal) {
    MEDI_UNUSED(value);
    MEDI_UNUSED(oldPrimal);
    // do nothing value should have an index

    return value.loc();
  }

  static bool isActive() {
    return isTaping();
  }
};
