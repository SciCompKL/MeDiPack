element: definitions
  root element
  contains elements: define, datatype, operator, typedef

element: define
  child of: definitions
  attributes:            name -> The name of the definition
                      version -> The mpi version this definition was added to the standard.
        [optional] deprecated -> The mpi version this definition was removed from the standard.
           [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.

element: datatype
  child of: definitions
  attributes:            name -> The name of the datatype
                         type -> The type of the datatype
                      version -> The mpi version this type was added to the standard.
        [optional] deprecated -> The mpi version this definition was removed from the standard.
           [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.

element: operator
  child of: definitions
  attributes:            name -> The name of the operator
                      version -> The mpi version this operator was added to the standard.
        [optional] deprecated -> The mpi version this definition was removed from the standard.
           [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.

element: typedef
  child of: definitions
  attributes:            name -> The name of the typedef
                      version -> The mpi version this typedef was added to the standard.
        [optional] deprecated -> The mpi version this definition was removed from the standard.
           [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.
