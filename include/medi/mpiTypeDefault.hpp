#pragma once

#include "macros.h"
#include "mpiTypeInterface.hpp"
#include "mpiOp.hpp"

namespace medi {

  template<typename ADTool>
  class MpiTypeDefault final
      : public MpiTypeBase<
          MpiTypeDefault<ADTool>,
          typename ADTool::Type,
          typename ADTool::ModifiedType,
          ADTool>
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
        MpiTypeBase<MpiTypeDefault<ADTool>, Type, ModifiedType, Tool>(Tool::MpiType, Tool::ModifiedMpiType),
        adTool(Tool::AdjointMpiType) {}

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

      inline void performReduce(Type* buf, Type* target, int count, AMPI_Op op, int ranks) const {
        for(int j = 1; j < ranks; ++j) {
          MPI_Reduce_local(&buf[j * count], buf, count, this->getMpiType(), op.primalFunction);
        }

        copy(buf, 0, target, 0, count);
      }

      void copy(Type* from, size_t fromOffset, Type* to, size_t toOffset, int count) const {
        for(int i = 0; i < count; ++i) {
          to[toOffset + i] = from[fromOffset + i];
        }

      }

      inline void createTypeBuffer(Type* &buf, size_t size) const {
        buf = new Type[size];
      }

      inline void createModifiedTypeBuffer(ModifiedType* &buf, size_t size) const {
        buf = new ModifiedType[size];
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
  };
}
