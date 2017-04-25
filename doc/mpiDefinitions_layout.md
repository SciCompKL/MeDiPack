element: definitions
  root element
	contains elements: define, datatype, operator, typedef

element: define
	child of: definitions
	attributes:            name -> The name of the definition
					 [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.

element: define
	child of: datatype
	attributes:            name -> The name of the datatype
	                       type -> The type of the datatype
					 [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.

element: operator
	child of: definitions
	attributes:            name -> The name of the operator
					 [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.

element: typedef
	child of: definitions
	attributes:            name -> The name of the typedef
					 [optional] special -> Indicates special handling of the type and generation will be ignored.
                                 The value gives the file name wehre the special handling is performed.
