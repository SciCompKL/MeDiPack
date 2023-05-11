cmake_minimum_required(VERSION 3.12)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/medipack-include.cmake)

project(
  ${MEDIPACK_NAME}
  VERSION ${MEDIPACK_VERSION}
  DESCRIPTION "MeDiPack - Message Differentiation Package"
  HOMEPAGE_URL "http://www.scicomp.uni-kl.de/software/medi")

set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(MPI REQUIRED)

include(GNUInstallDirs)

add_library(${PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/medi/medi.cpp)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  MPI::MPI_CXX)

target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_options(${PROJECT_NAME} INTERFACE "-DMEDI_HeaderOnly=0")

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION})

install(TARGETS ${PROJECT_NAME} EXPORT medipack-config
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    medipack-config-version.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(EXPORT medipack-config DESTINATION share/MeDiPack/cmake)
export(TARGETS ${PROJECT_NAME} FILE medipack-config.cmake)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/medipack-config-version.cmake" DESTINATION share/MeDiPack/cmake)
