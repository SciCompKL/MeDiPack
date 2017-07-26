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

#include "medipack.h"

#include "adToolInterface.h"
#include "mpiTypeDefault.hpp"

template <typename T>
void codiModifiedDependency(T& inval, T& inoutval) {

  bool active = (0 != inoutval.getGradientData()) | (0 != inval.getGradientData());
  if(active) {
    inoutval.getGradientData() = -1; // TODO: Define invalid index in CoDiPack
  } else {
    inoutval.getGradientData() = 0; // TODO: Define passive index in CoDiPack
  }
}

template <typename T>
void codiModifiedAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() += invec[i].value();
    codiModifiedDependency(invec[i], inoutvec[i]);
  }
}

template <typename T>
void codiModifiedMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() *= invec[i].value();
    codiModifiedDependency(invec[i], inoutvec[i]);
  }
}

template <typename T>
void codiModifiedMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() = max(inoutvec[i].value(), invec[i].value());
    codiModifiedDependency(invec[i], inoutvec[i]);
  }
}

template <typename T>
void codiModifiedMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() = min(inoutvec[i].value(), invec[i].value());
    codiModifiedDependency(invec[i], inoutvec[i]);
  }
}

template <typename T>
void codiUnmodifiedAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] += invec[i];
  }
}

template <typename T>
void codiUnmodifiedMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] *= invec[i];
  }
}

template <typename T>
void codiUnmodifiedMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = max(inoutvec[i], invec[i]);
  }
}

template <typename T>
void codiUnmodifiedMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = min(inoutvec[i], invec[i]);
  }
}

template <typename AT, typename PT>
void codiPreAdjMul(AT* adjoints, PT* primals, int count) {
  for(int i = 0; i < count; ++i) {
    adjoints[i] *= primals[i];
  }
}

template <typename AT, typename PT>
void codiPostAdjMul(AT* adjoints, PT* primals, PT* rootPrimals, int count) {
  CODI_UNUSED(rootPrimals);

  for(int i = 0; i < count; ++i) {
    if(0.0 != primals[i]) {
      adjoints[i] /= primals[i];
    }
  }
}

template <typename AT, typename PT>
void codiPostAdjMinMax(AT* adjoints, PT* primals, PT* rootPrimals, int count) {
  for(int i = 0; i < count; ++i) {
    if(rootPrimals[i] != primals[i]) {
      adjoints[i] = 0.0; // the primal of this process was not the minimum or maximum so do not perfrom the adjoint update
    }
  }
}

template<typename CoDiType, bool primalRestore, typename Impl>
struct CoDiPackToolBase : public medi::ADToolBase<Impl, typename CoDiType::GradientValue, typename CoDiType::PassiveReal, typename CoDiType::GradientData> {
  typedef CoDiType Type;
  typedef typename CoDiType::TapeType Tape;
  typedef typename CoDiType::GradientValue AdjointType;
  typedef CoDiType ModifiedType;
  typedef typename CoDiType::PassiveReal PassiveType;
  typedef typename CoDiType::GradientData IndexType;

  const static bool IS_ActiveType = true;
  const static bool IS_RequiresModifiedBuffer = false;

  static MPI_Datatype MpiType;
  static MPI_Datatype ModifiedMpiType;
  static MPI_Datatype AdjointMpiType;
  static medi::AMPI_Op OP_SUM;
  static medi::AMPI_Op OP_PROD;
  static medi::AMPI_Op OP_MIN;
  static medi::AMPI_Op OP_MAX;

  typedef medi::MpiTypeDefault<Impl> MediType;
  static MediType* MPI_TYPE;

  static Tape* adjointTape;

  CoDiPackToolBase(MPI_Datatype adjointMpiType) :
    medi::ADToolBase<Impl, typename CoDiType::GradientValue, typename CoDiType::PassiveReal, typename CoDiType::GradientData>(adjointMpiType) {}

  static void initTypes() {
    // create the mpi type for CoDiPack
    // this type is used in this type and the passive formulation
    // TODO: add proper type creation
    MPI_Type_contiguous(sizeof(CoDiType), MPI_BYTE, &MpiType);
    MPI_Type_commit(&MpiType);

    ModifiedMpiType = MpiType;
    AdjointMpiType = MPI_DOUBLE;
  }

  static void initOperator(medi::AMPI_Op& op, bool requiresPrimal, bool requiresPrimalSend, MPI_User_function* modifiedFunc, MPI_User_function* primalFunc, const medi::PreAdjointOperation preAdjointOperation, const medi::PostAdjointOperation postAdjointOperation) {
    op.init(requiresPrimal, requiresPrimalSend, primalFunc, true, modifiedFunc, true, preAdjointOperation, postAdjointOperation);
  }

  static void initOperator(medi::AMPI_Op& op, MPI_User_function* primalFunc) {
    op.init(primalFunc, true);
  }

  static void initOperators() {
    initOperator(OP_SUM, false, false, (MPI_User_function*)codiModifiedAdd<Type>, (MPI_User_function*)codiUnmodifiedAdd<Type>, medi::noPreAdjointOperation, medi::noPostAdjointOperation);
    initOperator(OP_PROD, (MPI_User_function*)codiUnmodifiedMul<Type>);
    initOperator(OP_MIN, true, true, (MPI_User_function*)codiModifiedMin<Type>, (MPI_User_function*)codiUnmodifiedMin<Type>, medi::noPreAdjointOperation, (medi::PostAdjointOperation)codiPostAdjMinMax<double, double>);
    initOperator(OP_MAX, true, true, (MPI_User_function*)codiModifiedMax<Type>, (MPI_User_function*)codiUnmodifiedMax<Type>, medi::noPreAdjointOperation, (medi::PostAdjointOperation)codiPostAdjMinMax<double, double>);
  }

  static void init() {
    initTypes();
    initOperators();

    MPI_TYPE = new MediType();
  }

  static void finalizeOperators() {
    OP_SUM.free();
    OP_PROD.free();
    OP_MIN.free();
    OP_MAX.free();
  }

  static void finalizeTypes() {
    MPI_Type_free(&MpiType);
  }

  static void finalize() {
    if(nullptr != MPI_TYPE) {
      delete MPI_TYPE;
      MPI_TYPE = nullptr;
    }

    finalizeOperators();
    finalizeTypes();
  }

  inline bool isActiveType() const {
    return true;
  }

  inline  bool isHandleRequired() const {
    return Type::getGlobalTape().isActive();
  }

  inline bool isOldPrimalsRequired() const {
    return primalRestore;
  }

  inline void startAssembly(medi::HandleBase* h) {
    MEDI_UNUSED(h);

  }

  inline void addToolAction(medi::HandleBase* h) {
    if(NULL != h) {
      Type::getGlobalTape().pushExternalFunctionHandle(callFunc, h, deleteFunc);
    }
  }

  inline void stopAssembly(medi::HandleBase* h) {
    MEDI_UNUSED(h);
  }

  inline void getAdjoints(const IndexType* indices, AdjointType* adjoints, int elements) const {
    for(int pos = 0; pos < elements; ++pos) {
      IndexType index = indices[pos];
      AdjointType& grad = adjointTape->gradient(index);
      adjoints[pos] = grad;
      grad = 0.0;
    }
  }

  inline void updateAdjoints(const IndexType* indices, const AdjointType* adjoints, int elements) const {
    for(int pos = 0; pos < elements; ++pos) {
      IndexType indexCopy = indices[pos];
      adjointTape->gradient(indexCopy) += adjoints[pos];
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

  static void callFunc(void* tape, void* h) {
    adjointTape = (Tape*)tape;
    medi::HandleBase* handle = static_cast<medi::HandleBase*>(h);
    handle->func(handle);
  }

  static void deleteFunc(void* tape, void* h) {
    MEDI_UNUSED(tape);

    medi::HandleBase* handle = static_cast<medi::HandleBase*>(h);
    delete handle;
  }

  static inline IndexType getIndex(const Type& value) {
    return value.getGradientData();
  }

  static inline void clearIndex(Type& value) {
    IndexType oldIndex = value.getGradientData();
    value.~Type();
    value.getGradientData() = oldIndex;  // restore the index here so that the other side can decide of the communication was active or not
  }

  static inline PassiveType getValue(const Type& value) {
    return value.getValue();
  }

  static inline void setValue(const IndexType& index, const PassiveType& primal) {
    MEDI_UNUSED(index);
    MEDI_UNUSED(primal);
  }

  static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) {
    MEDI_UNUSED(modValue);
    MEDI_UNUSED(value);
  }

  static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) {
    MEDI_UNUSED(modValue);
    if(0 != value.getGradientData()) {
      value.getGradientData() = IndexType();
      Type::getGlobalTape().registerInput(value);
    }
  }

  static bool isActive() {
    return Type::getGlobalTape().isActive();
  }
};

template<typename CoDiType>
struct CoDiPackTool final : public CoDiPackToolBase<CoDiType, false, CoDiPackTool<CoDiType> >  {

    typedef CoDiType Type;
    typedef typename CoDiType::TapeType Tape;
    typedef typename CoDiType::GradientValue AdjointType;
    typedef CoDiType ModifiedType;
    typedef typename CoDiType::PassiveReal PassiveType;
    typedef typename CoDiType::GradientData IndexType;

    CoDiPackTool(MPI_Datatype adjointMpiType) :
      CoDiPackToolBase<CoDiType, false, CoDiPackTool< CoDiType>>(adjointMpiType) {}


    inline void setReverseValues(const IndexType* indices, const PassiveType* primals, int elements) const {
      MEDI_UNUSED(indices);
      MEDI_UNUSED(primals);
      MEDI_UNUSED(elements);

      //Do nothing
    }

    static inline IndexType registerValue(Type& value, PassiveType& oldPrimal) {
      MEDI_UNUSED(oldPrimal);

      bool wasActive = 0 != value.getGradientData();
      value.getGradientData() = IndexType();

      // make the value active again if it has been active before on the other processor
      if(wasActive) {
        Type::getGlobalTape().registerInput(value);
      }

      return value.getGradientData();
    }
};

template<typename CoDiType>
struct CoDiPackToolPrimalRestore final : public CoDiPackToolBase<CoDiType, true, CoDiPackToolPrimalRestore<CoDiType> >  {

    typedef CoDiType Type;
    typedef typename CoDiType::TapeType Tape;
    typedef typename CoDiType::GradientValue AdjointType;
    typedef CoDiType ModifiedType;
    typedef typename CoDiType::PassiveReal PassiveType;
    typedef typename CoDiType::GradientData IndexType;

    CoDiPackToolPrimalRestore(MPI_Datatype adjointMpiType) :
      CoDiPackToolBase<CoDiType, true, CoDiPackToolPrimalRestore< CoDiType>>(adjointMpiType) {}

    inline void setReverseValues(const IndexType* indices, const PassiveType* primals, int elements) const {
      MEDI_UNUSED(indices);
      MEDI_UNUSED(primals);
      MEDI_UNUSED(elements);

      for(int pos = 0; pos < elements; ++pos) {
        CoDiPackToolBase<CoDiType, true, CoDiPackToolPrimalRestore<CoDiType> >::adjointTape->setExternalValueChange(indices[pos], primals[pos]);
      }
    }

    static inline IndexType registerValue(Type& value, PassiveType& oldPrimal) {
      bool wasActive = 0 != value.getGradientData();
      value.getGradientData() = IndexType();

      if(wasActive) {
        oldPrimal = Type::getGlobalTape().registerExtFunctionOutput(value);
      } else {
        oldPrimal = 0.0;
      }

      return value.getGradientData();
    }
};

template<typename CoDiType, bool primalRestore, typename Impl> MPI_Datatype CoDiPackToolBase<CoDiType, primalRestore, Impl>::MpiType;
template<typename CoDiType, bool primalRestore, typename Impl> MPI_Datatype CoDiPackToolBase<CoDiType, primalRestore, Impl>::ModifiedMpiType;
template<typename CoDiType, bool primalRestore, typename Impl> MPI_Datatype CoDiPackToolBase<CoDiType, primalRestore, Impl>::AdjointMpiType;
template<typename CoDiType, bool primalRestore, typename Impl> medi::AMPI_Op CoDiPackToolBase<CoDiType, primalRestore, Impl>::OP_SUM;
template<typename CoDiType, bool primalRestore, typename Impl> medi::AMPI_Op CoDiPackToolBase<CoDiType, primalRestore, Impl>::OP_PROD;
template<typename CoDiType, bool primalRestore, typename Impl> medi::AMPI_Op CoDiPackToolBase<CoDiType, primalRestore, Impl>::OP_MIN;
template<typename CoDiType, bool primalRestore, typename Impl> medi::AMPI_Op CoDiPackToolBase<CoDiType, primalRestore, Impl>::OP_MAX;
template<typename CoDiType, bool primalRestore, typename Impl> typename CoDiPackToolBase<CoDiType, primalRestore, Impl>::MediType* CoDiPackToolBase<CoDiType, primalRestore, Impl>::MPI_TYPE;
template<typename CoDiType, bool primalRestore, typename Impl> typename CoDiType::TapeType* CoDiPackToolBase<CoDiType, primalRestore, Impl>::adjointTape;
