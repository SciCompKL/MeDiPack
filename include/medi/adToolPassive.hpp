#pragma once

#include "adToolInterface.h"
#include "macros.h"
#include "typeDefinitions.h"

namespace medi {

  /**
   * @brief The interface for the AD tool that is accessed by MeDiPack.
   */
  class ADToolPassive final : public ADToolBase<ADToolPassive, void, void, void> {
    public:

      typedef void PassiveType;
      typedef void AdjointType;
      typedef void IndexType;

      ADToolPassive(MPI_Datatype adjointType) :
        ADToolBase<ADToolPassive, void, void, void>(adjointType)
      {}

      inline bool isActiveType() const {return false;}
      inline bool isHandleRequired() const {return false;}
      inline bool isOldPrimalsRequired() const {return false;}
      inline void startAssembly(HandleBase* h) {MEDI_UNUSED(h);}
      inline void stopAssembly(HandleBase* h) {MEDI_UNUSED(h);}
      inline void addToolAction(HandleBase* h) {MEDI_UNUSED(h);}

      inline void getAdjoints(const IndexType* indices, AdjointType* adjoints, int elements) const {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(adjoints);
        MEDI_UNUSED(elements);
      }

      inline void updateAdjoints(const IndexType* indices, const AdjointType* adjoints, int elements) const {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(adjoints);
        MEDI_UNUSED(elements);
      }

      inline void setReverseValues(const IndexType* indices, const PassiveType* primals, int elements) const {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(primals);
        MEDI_UNUSED(elements);
      }

      inline void combineAdjoints(AdjointType* buf, const int elements, const int ranks) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(elements);
        MEDI_UNUSED(ranks);
      }

      inline void createAdjointTypeBuffer(AdjointType* &buf, size_t size) const {
        MEDI_UNUSED(size);

        buf = nullptr;
      }

      inline void createPassiveTypeBuffer(PassiveType* &buf, size_t size) const {
        MEDI_UNUSED(size);

        buf = nullptr;
      }

      inline void createIndexTypeBuffer(IndexType* &buf, size_t size) const {
        MEDI_UNUSED(size);

        buf = nullptr;
      }

      inline void deleteAdjointTypeBuffer(AdjointType* &buf) const {
        buf = nullptr;
      }

      inline void deletePassiveTypeBuffer(PassiveType* &buf) const {
        buf = nullptr;
      }

      inline void deleteIndexTypeBuffer(IndexType* &buf) const {
        buf = nullptr;
      }
  };
}
