element: functions
  root element
  contains elements: function

element: function
  child of: functions
  contains elements: arg, send, recv, displs, operator, request, type, message, status
  attributes:            name -> The name of the function. Used for naming handles, functions etc.
                      version -> The mpi version this function was added to the standard.
        [optional] deprecated -> The mpi version this definition was removed from the standard.
             [optional] async -> Defines that the function is asynchronous. The value defines the name of the
                                 argument that is the mpi request.
             [optional]  init -> Adds additional splits for persistent requests. Implies that async is also set.
                                 The value defines the function to which the recorded requests are wrapped.
              [optional] type -> The return type of the function. (Default: int)
            [optinal] mpiName -> Defines the name that is used for the real mpi function in the generation process. If not defined the name of the function will be taken.
         [optinal] mediHandle -> Defines the handling type of MeDiPack for this function possible options are:
                                    transform: Preforms AD handling
                                      disable: Ignores the function
                                      handled: Ignores the function, but indicates that the function is handled in a wrapper.
                                      forward: Creates a define for the function forwarding.
                                       ignore: Ignores the function because it is not required.
                                      default: Just produce a forward handling to the normal MPI function. [default value]


element: arg
  -> A regular argument that does not need special handling.
  child of: function
  attributes:             name -> The name of the argument.
                          type -> The type of the argument.
              [optional] const -> If defined indicates that the argument is constant. If set to opt the constant modifier is generated as optional.

element: send/recv
  -> The send and recv elements define send and recieve buffers of the functions.
  child of: function
  attributes:                    name -> The name of the buffer.
                                 type -> The name of the data type that defines the type for this buffer
                                count -> The name of the argument that gives the count for this buffer
              [optional] ranks,displs -> Indicate different layouts fo the buffer. Only one allowed.
                                  ranks -> Indicates that the forward buffer spans all ranks. E.g. Allgather
                                           The value defines the name of the communicator.
                                 displs -> Indicates that the buffer is defined by counts and displacements and switches
                                           therefore the generated code for the buffer.
                                           The value indicates the name of the displacement vector. The count attribute defines
                                           the name of the counts.
                       [optional] all -> Indicates that the buffer needs to span all ranks for the reverse operation. E.g. Allgather
                                         The value defines the name of the communicator.
                      [optional] root -> Indicates that the buffer only existas at the root process. The values defines the name
                                         argument that gives the root number.
                     [optional] const -> If defined indicates that the argument is constant. If set to opt the constant modifier is generated as optional.
                   [optional] inplace -> Indicates that the buffer can be MPI_IN_PLACE. The value of the field indicates
                                         name of the buffer that contains the data.

element: displs
  -> Special handling for displacement arguments. They need to be stored in the handle with additional information.
  child of: function
  attributes:             name -> The name of the argument.
                          type -> The type of the argument.
              [optional] const -> If defined indicates that the argument is constant. If set to opt the constant modifier is generated as optional.

element: operator
  -> Special handling for operator arguments. The primal values may need to be stored for the reverse operation.
  child of: function
  attributes:             name -> The name of the argument.
                          type -> The type of the argument.
              [optional] const -> If defined indicates that the argument is constant. If set to opt the constant modifier is generated as optional.

element: request
  -> Special handling for request arguments. A special request for the reverse call needs to be created.
  child of: function
  attributes:             name -> The name of the argument.
                          type -> The type of the argument.
              [optional] const -> If defined indicates that the argument is constant. If set to opt the constant modifier is generated as optional.
              [optional] noptr -> Indicates that the request is no pointer type.

element: type
  -> Special handling for datatype arguments. Depending on the interface they are treated as templates or normal arguments.
  child of: function
  attributes:             name -> The name of the argument.
                          type -> The type of the argument.
              [optional] const -> If defined indicates that the argument is constant. If set to opt the constant modifier is generated as optional.

element: message
  -> Special handling for message arguments. They are directly stored in the buffers.
  child of: function
  attributes:             name -> The name of the argument.
                          type -> The type of the argument.
              [optional] const -> If defined indicates that the argument is constant. If set to opt the constant modifier is generated as optional.

element: status
  -> Special handling such that it is not stored and created for the reverse calls.
  child of: function
  attributes:             name -> The name of the argument.
                          type -> The type of the argument.
