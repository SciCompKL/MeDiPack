#pragma once

#include "macros.h"
#include "mpiTypeInterface.hpp"
#include "op.hpp"

namespace medi {

  template<typename ADTool>
  class MpiTypeDefault
      : public MpiTypeBase<
          MpiTypeDefault<ADTool>,
          typename ADTool::Type,
          typename ADTool::ModifiedType,
          typename ADTool::PassiveType,
          typename ADTool::AdjointType,
          typename ADTool::IndexType>
  {

    public:

      typedef typename ADTool::Type Type;
      typedef typename ADTool::ModifiedType ModifiedType;
      typedef typename ADTool::PassiveType PassiveType;
      typedef typename ADTool::AdjointType AdjointType;
      typedef typename ADTool::IndexType IndexType;

      typedef ADTool Tool;

      Tool adTool;

      MpiTypeDefault() :
        MpiTypeBase<MpiTypeDefault<ADTool>, Type, ModifiedType, PassiveType, AdjointType, IndexType>(Tool::MPIType),
        adTool() {}

      Tool& getADTool() {
        return adTool;
      }

      const int getValuesPerElement(const Type* buf) const {
        MEDI_UNUSED(buf);

        return 1;
      }

      bool isModifiedBufferRequired() const {
        return Tool::IS_RequiresModifiedBuffer;
      }

      inline void copyIntoModifiedBuffer(const Type* buf, size_t bufOffset, ModifiedType* bufMod, size_t bufModOffset, int elements) const {
        if(ADTool::IS_RequiresModifiedBuffer) {
          for(int i = 0; i < elements; ++i) {
            ADTool::setIntoModifyBuffer(bufMod[bufModOffset + i], buf[bufOffset + i]);
          }
        }
      }

      inline void copyFromModifiedBuffer(Type* buf, size_t bufOffset, const ModifiedType* bufMod, size_t bufModOffset, int elements) const {
        if(ADTool::IS_RequiresModifiedBuffer) {
          for(int i = 0; i < elements; ++i) {
            ADTool::getFromModifyBuffer(bufMod[bufModOffset + i], buf[bufOffset + i]);
          }
        }
      }

      inline void getIndices(const Type* buf, size_t bufOffset, IndexType* indices, size_t indexOffset, int elements) const {
        for(int i = 0; i < elements; ++i) {
          indices[indexOffset + i] = ADTool::getIndex(buf[bufOffset + i]);
        }
      }

      inline void registerValue(Type* buf, size_t bufOffset, IndexType* indices, size_t indexOffset, int elements) const {
        for(int i = 0; i < elements; ++i) {
          indices[indexOffset + i] = ADTool::registerValue(buf[bufOffset + i]);;
        }
      }

      inline void clearIndices(Type* buf, size_t bufOffset, int elements) const {
        for(int i = 0; i < elements; ++i) {
          ADTool::clearIndex(buf[bufOffset + i]);
        }
      }

      inline void getValues(const Type* buf, size_t bufOffset, PassiveType* primals, size_t primalOffset, int elements) const {
        for(int pos = 0; pos < elements; ++pos) {
          primals[primalOffset + pos] = ADTool::getValue(buf[bufOffset + pos]);
        }
      }

      inline void performReduce(Type* buf, Type* target, int count, TAMPI_Op op, int ranks) const {
        for(int j = 1; j < ranks; ++j) {
          MPI_Reduce_local(&buf[j * count], buf, count, this->getMpiType(), op.primalFunction);
        }

        for(int i = 0; i < count; ++i) {
          target[i] = buf[i];
        }
      }

      inline void getAdjoints(const IndexType* indices, AdjointType* adjoints, int elements) const {
        for(int pos = 0; pos < elements; ++pos) {
          adjoints[pos] = ADTool::getAdjoint(indices[pos]);
        }
      }

      inline void updateAdjoints(const IndexType* indices, const AdjointType* adjoints, int elements) const {
        for(int pos = 0; pos < elements; ++pos) {
          ADTool::updateAdjoint(indices[pos], adjoints[pos]);
        }
      }

      inline void combineAdjoints(AdjointType* buf, const int elements, const int ranks) const {
        for(int curRank = 1; curRank < ranks; ++curRank) {
          for(int curPos = 0; curPos < elements; ++curPos) {
            buf[curPos] += buf[elements * curRank + curPos];
          }
        }
      }

      inline void createTypeBuffer(Type* &buf, size_t size) const {
        buf = new Type[size];
      }

      inline void createModifiedTypeBuffer(ModifiedType* &buf, size_t size) const {
        buf = new ModifiedType[size];
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

      inline void deleteTypeBuffer(Type* &buf) const {
        if(NULL != buf) {
          delete [] buf;
          buf = NULL;
        }
      }

      inline void deleteModifiedTypeBuffer(ModifiedType* &buf) const {
        if(NULL != buf) {
          delete [] buf;
          buf = NULL;
        }
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
  };
}
