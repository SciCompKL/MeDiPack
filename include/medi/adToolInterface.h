#pragma once

#include "typeDefinitions.h"

namespace medi {

  /**
   * @brief The interface for the AD tool that is accessed by MeDiPack.
   */
  class ADToolInterface {
    public:
      virtual bool isActiveType() const = 0;
      virtual bool isHandleRequired() const  = 0;
      virtual bool isOldPrimalsRequired() const = 0;
      virtual void startAssembly(HandleBase* h) = 0;
      virtual void stopAssembly(HandleBase* h) = 0;
      virtual void addToolAction(HandleBase* h) = 0;
  };
}
