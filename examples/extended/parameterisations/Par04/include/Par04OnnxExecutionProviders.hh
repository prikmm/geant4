// add oneDNN/DNNL execution provider specific header
#ifdef USE_DNNL
#include "dnnl_provider_factory.h"
#endif

// add OpenVINO execution provider specific header
#ifdef USE_OPENVINO
#include "dnnl_provider_factory.h"
#endif

// add TensorRT execution provider specific header
#ifdef USE_TENSORRT
#include "dnnl_provider_factory.h"
#include "dnnl_provider_factory.h"
#endif