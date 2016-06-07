#pragma once

#include "../medipack.h"

namespace medi {

  struct TAMPI_Request {
      MPI_Request request;
      HandleBase* handle;
      ContinueFunction func;
  };

  inline void TAMPI_Wait_b(HandleBase* handle);
  struct WaitHandle : HandleBase {
      ReverseFunction finishFunc;
      HandleBase* handle;

      WaitHandle(ReverseFunction finishFunc, HandleBase* handle) :
        finishFunc(finishFunc),
        handle(handle) {
         this->func = (ReverseFunction)TAMPI_Wait_b;
      }
  };

  inline int TAMPI_Wait(TAMPI_Request *request, TAMPI_Status *status) {
    if(TAMPI_REQUEST_NULL == request) {
      return 0;
    }

    int rStatus = MPI_Wait(&request->request, status);

    request->func(request->handle);

    return rStatus;
  }


  inline void TAMPI_Wait_b(HandleBase* handle) {
    WaitHandle* h = static_cast<WaitHandle*>(handle);

    h->finishFunc(h->handle);
  }
}
