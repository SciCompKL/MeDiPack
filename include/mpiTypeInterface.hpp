#pragma once

#include <mpi.h>

#include "adToolInterface.h"

namespace medi {
  class MpiTypeInterface {
    private:

      MPI_Datatype mpiType;
    public:

      typedef void Type;
      typedef void ModifiedType;
      typedef void AdjointType;
      typedef void PassiveType;
      typedef void IndexType;

      MpiTypeInterface(MPI_Datatype mpiType) :
        mpiType(mpiType) {}

      MPI_Datatype getMpiType() const {
        return mpiType;
      }

      virtual bool isModifiedBufferRequired() const = 0;

      virtual int getValuesPerElement(const void* buf) const = 0;

      virtual ADToolInterface& getADTool() = 0;

      virtual void copyIntoModifiedBuffer(const void* buf, void* bufMod, int elements) const = 0;

      virtual void copyFromModifiedBuffer(void* buf, const void* bufMod, int elements) const = 0;

      virtual void getIndices(const void* buf, void* indices, int elements) const = 0;

      virtual void registerValue(void* buf, void* indices, int elements) const = 0;

      virtual void clearIndices(void* buf, int elements) const = 0;

      virtual void getValues(const void* buf, void* primals, int elements) const = 0;

      virtual void getAdjoints(const void* indices, int count, void* adjoints) const = 0;

      virtual void updateAdjoints(const void* indices, int count, const void* adjoints) const = 0;
  };

  /**
   * Implements all functions from MpiType that require a type change.
   *
   * Functions that are not implemented: isModifiedBufferRequired
   */
  template <typename Impl, typename TypeB, typename ModifiedTypeB, typename PassiveTypeB, typename AdjointTypeB, typename IndexTypeB>
  class MpiTypeBase : public MpiTypeInterface {
    public:

      MpiTypeBase(MPI_Datatype mpiType) :
        MpiTypeInterface(mpiType) {}

      int getValuesPerElement(const void* buf) const {
        return cast().getValuesPerElement(castBuffer<TypeB>(buf));
      }

      void copyIntoModifiedBuffer(const void* buf, void* bufMod, int elements) const {
        cast().copyIntoModifiedBuffer(castBuffer<TypeB>(buf), castBuffer<ModifiedTypeB>(bufMod), elements);
      }

      void copyFromModifiedBuffer(void* buf, const void* bufMod, int elements) const {
        cast().copyFromModifiedBuffer(castBuffer<TypeB>(buf), castBuffer<ModifiedTypeB>(bufMod), elements);
      }

      void getIndices(const void* buf, void* indices, int elements) const {
        cast().getIndices(castBuffer<TypeB>(buf), castBuffer<IndexTypeB>(indices), elements);
      }

      void registerValue(void* buf, void* indices, int elements) const {
        cast().registerValue(castBuffer<TypeB>(buf), castBuffer<IndexTypeB>(indices), elements);
      }

      void clearIndices(void* buf, int elements) const {
        cast().clearIndices(castBuffer<TypeB>(buf), elements);
      }

      void getValues(const void* buf, void* primals, int elements) const {
        cast().getValues(castBuffer<TypeB>(buf), castBuffer<PassiveTypeB>(primals), elements);
      }

      void getAdjoints(const void* indices, int count, void* adjoints) const {
        cast().getAdjoints(castBuffer<IndexTypeB>(indices), count, castBuffer<AdjointTypeB>(adjoints));
      }

      void updateAdjoints(const void* indices, int count, const void* adjoints) const {
        cast().updateAdjoints(castBuffer<IndexTypeB>(indices), count, castBuffer<AdjointTypeB>(adjoints));
      }

    private:

      inline Impl& cast() {
        return *reinterpret_cast<Impl*>(this);
      }

      inline const Impl& cast() const {
        return *reinterpret_cast<const Impl*>(this);
      }

      template <typename T>
      inline T* castBuffer(void* buf) const {
        return reinterpret_cast<T*>(buf);
      }

      template <typename T>
      inline const T* castBuffer(const void* buf) const {
        return reinterpret_cast<const T*>(buf);
      }
  };
}


