# Find the ONNX Runtime include directory and library.
#
# This module defines the `onnxruntime` imported target that encodes all
# necessary information in its target properties.

find_library(
  OnnxRuntime_LIBRARY
  NAMES onnxruntime
  PATH_SUFFIXES lib lib32 lib64
  DOC "The ONNXRuntime library")

find_library(
  OnnxRuntime_SHARED_lib
  NAMES onnxruntime_providers_shared
  PATH_SUFFIXES lib lib32 lib64
  DOC "The ONNXRuntime EP shared library")

find_library(
  OnnxRuntime_DNNL_lib
  NAMES onnxruntime_providers_dnnl
  PATH_SUFFIXES lib lib32 lib64
  DOC "The ONNXRuntime DNNL EP shared library")

find_library(
  OnnxRuntime_OPENVINO_lib
  NAMES onnxruntime_providers_openvino
  PATH_SUFFIXES lib lib32 lib64
  DOC "The ONNXRuntime OpenVINO EP shared library")

find_library(
  OnnxRuntime_TENSORRT_lib
  NAMES onnxruntime_providers_tensorrt
  PATH_SUFFIXES lib lib32 lib64
  DOC "The ONNXRuntime TensorRT EP shared library")

find_path(
  OnnxRuntime_CXX_INCLUDE
  NAMES onnxruntime_cxx_api.h
  PATH_SUFFIXES include/onnxruntime/core/session
  DOC "The ONNXRuntime C++ include directory")

find_path(
  OnnxRuntime_C_INCLUDE
  NAMES onnxruntime_c_api.h
  PATH_SUFFIXES include/onnxruntime/core/session
  DOC "The ONNXRuntime C include directory")

find_path(
  OnnxRuntime_DNNL_EP_INCLUDE
  NAMES dnnl_provider_factory.h 
  PATH_SUFFIXES include/onnxruntime/core/providers/dnnl
  DOC "The ONNXRuntime DNNL provider include directory")

find_path(
  OnnxRuntime_OPENVINO_EP_INCLUDE
  NAMES openvino_provider_factory.h 
  PATH_SUFFIXES include/onnxruntime/core/providers/openvino
  DOC "The ONNXRuntime OpenVINO provider include directory")

find_path(
  OnnxRuntime_TENSORRT_EP_INCLUDE_factory
  NAMES tensorrt_provider_factory.h 
  PATH_SUFFIXES include/onnxruntime/core/providers/tensorrt
  DOC "The ONNXRuntime TensorRT provider include directory")

find_path(
  OnnxRuntime_TENSORRT_EP_INCLUDE_options
  NAMES core/providers/tensorrt/tensorrt_provider_options.h 
  PATH_SUFFIXES include include/onnxruntime
  DOC "The ONNXRuntime TensorRT provider include directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OnnxRuntime
  REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_CXX_INCLUDE OnnxRuntime_C_INCLUDE)

find_package_handle_standard_args(
  OnnxRuntime_DnnlEp
  REQUIRED_VARS OnnxRuntime_DNNL_EP_INCLUDE)

find_package_handle_standard_args(
  OnnxRuntime_OpenVinoEp
  REQUIRED_VARS OnnxRuntime_OPENVINO_EP_INCLUDE)
  
find_package_handle_standard_args(
  OnnxRuntime_TensorrtEp
  REQUIRED_VARS OnnxRuntime_TENSORRT_EP_INCLUDE_factory OnnxRuntime_TENSORRT_EP_INCLUDE_options)

find_package_handle_standard_args(
  OnnxRuntime_Shared_lib
  REQUIRED_VARS OnnxRuntime_SHARED_lib
)

add_library(OnnxRuntime SHARED IMPORTED)
set_property(TARGET OnnxRuntime PROPERTY IMPORTED_LOCATION ${OnnxRuntime_LIBRARY})
set_property(TARGET OnnxRuntime PROPERTY IMPORTED_LOCATION ${OnnxRuntime_DNNL_lib})
set_property(TARGET OnnxRuntime PROPERTY IMPORTED_LOCATION ${OnnxRuntime_OPENVINO_lib})
set_property(TARGET OnnxRuntime PROPERTY IMPORTED_LOCATION ${OnnxRuntime_TENSORRT_lib})
set_property(TARGET OnnxRuntime PROPERTY IMPORTED_LOCATION ${OnnxRuntime_SHARED_lib})
set_property(TARGET OnnxRuntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_CXX_INCLUDE})
set_property(TARGET OnnxRuntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_C_INCLUDE})
set_property(TARGET OnnxRuntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_DNNL_EP_INCLUDE})
set_property(TARGET OnnxRuntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_OPENVINO_EP_INCLUDE})
set_property(TARGET OnnxRuntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_TENSORRT_EP_INCLUDE_factory})
set_property(TARGET OnnxRuntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_TENSORRT_EP_INCLUDE_options})

mark_as_advanced(OnnxRuntime_FOUND OnnxRuntime_CXX_INCLUDE OnnxRuntime_C_INCLUDE OnnxRuntime_LIBRARY)
mark_as_advanced(OnnxRuntime_DnnlEp_FOUND OnnxRuntime_DNNL_EP_INCLUDE)
mark_as_advanced(OnnxRuntime_OpenVinoEp_FOUND OnnxRuntime_OPENVINO_EP_INCLUDE)
mark_as_advanced(OnnxRuntime_TensorrtEp_FOUND OnnxRuntime_TENSORRT_EP_INCLUDE_factory OnnxRuntime_TENSORRT_EP_INCLUDE_options)
mark_as_advanced(OnnxRuntime_Shared_lib_FOUND OnnxRuntime_SHARED_lib)





