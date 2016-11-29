#pragma once

#include <mpi.h>

#include "adToolPassive.hpp"
#include "macros.h"
#include "mpiTypeInterface.hpp"
#include "mpiOp.hpp"

namespace medi {

  template<typename T>
  class MpiTypePassive final
      : public MpiTypeBase<
          MpiTypePassive<T>,
          T,
          T,
          ADToolPassive>
  {

    public:

      typedef T Type;
      typedef T ModifiedType;
      typedef void PassiveType;
      typedef void IndexType;

      typedef ADToolPassive Tool;

      Tool adTool;

      MpiTypePassive(MPI_Datatype type) :
        MpiTypeBase<MpiTypePassive<T>, Type, ModifiedType, ADToolPassive>(type, type),
        adTool(type) {}

      Tool& getADTool() {
        return adTool;
      }

      int computeActiveElements(const int count) const {

        return count;
      }

      bool isModifiedBufferRequired() const {
        return false;
      }

      inline void copyIntoModifiedBuffer(const Type* buf, size_t bufOffset, ModifiedType* bufMod, size_t bufModOffset, int elements) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufOffset);
        MEDI_UNUSED(bufMod);
        MEDI_UNUSED(bufModOffset);
        MEDI_UNUSED(elements);
      }

      inline void copyFromModifiedBuffer(Type* buf, size_t bufOffset, const ModifiedType* bufMod, size_t bufModOffset, int elements) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufOffset);
        MEDI_UNUSED(bufMod);
        MEDI_UNUSED(bufModOffset);
        MEDI_UNUSED(elements);
      }

      inline void getIndices(const Type* buf, size_t bufOffset, IndexType* indices, size_t indexOffset, int elements) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufOffset);
        MEDI_UNUSED(indices);
        MEDI_UNUSED(indexOffset);
        MEDI_UNUSED(elements);
      }

      inline void registerValue(Type* buf, size_t bufOffset, IndexType* indices, size_t indexOffset, int elements) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufOffset);
        MEDI_UNUSED(indices);
        MEDI_UNUSED(indexOffset);
        MEDI_UNUSED(elements);
      }

      inline void clearIndices(Type* buf, size_t bufOffset, int elements) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufOffset);
        MEDI_UNUSED(elements);
      }

      inline void getValues(const Type* buf, size_t bufOffset, PassiveType* primals, size_t primalOffset, int elements) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufOffset);
        MEDI_UNUSED(primals);
        MEDI_UNUSED(primalOffset);
        MEDI_UNUSED(elements);
      }

      inline void performReduce(Type* buf, Type* target, int count, AMPI_Op op, int ranks) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(target);
        MEDI_UNUSED(count);
        MEDI_UNUSED(op);
        MEDI_UNUSED(ranks);
      }

      inline void copy(Type* from, size_t fromOffset, Type* to, size_t toOffset, int count) const {
        MEDI_UNUSED(from);
        MEDI_UNUSED(fromOffset);
        MEDI_UNUSED(to);
        MEDI_UNUSED(toOffset);
        MEDI_UNUSED(count);
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
