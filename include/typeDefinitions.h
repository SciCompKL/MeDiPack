#pragma once

namespace medi {

  struct HandleBase;
  typedef void (*ReverseFunction)(HandleBase* h);
  typedef void (*ContinueFunction)(HandleBase* h);
  typedef void (*PreAdjointOperation)(void* adjoints, void* primals, int count);
  typedef void (*PostAdjointOperation)(void* adjoints, void* primals, void* rootPrimals, int count);

  struct HandleBase {
    ReverseFunction func;

    virtual ~HandleBase() {}
  };

}
