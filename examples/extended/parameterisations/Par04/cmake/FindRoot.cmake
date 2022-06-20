# Find the ROOT include directory and library.
#
# This module defines the `root` imported target that encodes all
# necessary information in its target properties.
#
# This package is necessary for CPU memory profiling

find_library(
    Root_LIB
    NAMES Core
    PATH_SUFFIXES lib lib32 lib64
    DOC "Library required for CPU Memory usage info"
)

find_path(
    Root_INCLUDE
    NAMES TSystem.h
    PATH_SUFFIXES include
    DOC "Include directory required for CPU Memory usage info"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Root
    REQUIRED_VARS Root_LIB Root_INCLUDE
)

add_library(Root SHARED IMPORTED)
set_property(TARGET Root PROPERTY IMPORTED_LOCATION ${Root_LIB})
set_property(TARGET Root PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Root_INCLUDE})

mark_as_advanced(Root_FOUND Root_LIB Root_INCLUDE)
