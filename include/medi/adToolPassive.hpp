#pragma once

#include "adToolInterface.h"
#include "macros.h"
#include "typeDefinitions.h"

namespace medi {

  /**
   * @brief The interface for the AD tool that is accessed by MeDiPack.
   */
  class ADToolPassive final : public ADToolInterface {
    public:
      inline bool isActiveType() const {return false;}
      inline bool isHandleRequired() const {return false;}
      inline void startAssembly(HandleBase* h) {MEDI_UNUSED(h);}
      inline void stopAssembly(HandleBase* h) {MEDI_UNUSED(h);}
      inline void addToolAction(HandleBase* h) {MEDI_UNUSED(h);}
  };
}
