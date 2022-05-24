#include "session/onnxruntime_cxx_api.h"
#include "session/onnxruntime_c_api.h"
#include "providers/dnnl/dnnl_provider_factory.h"
#include "providers/openvino/openvino_provider_factory.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <string>

int main(){

    std::unique_ptr<Ort::Env> fEnv;
    /// Pointer to the ONNX inference session
    std::unique_ptr<Ort::Session> fSession;
    /// ONNX settings
    Ort::SessionOptions fSessionOptions;
    /// ONNX memory info
    const OrtMemoryInfo* fInfo;
    struct MemoryInfo;
    /// the input names represent the names given to the model
    /// when defining  the model's architecture (if applicable)
    /// they can also be retrieved from model.summary()
    std::vector<const char*> fInames;

    std::vector<float> aGenVector = {-1.98178,1.40776,0.286402,-1.04228,0.0167152,-0.0118188,0.403963,0.747496,-0.319002,-0.844152,0.00976546,0.999992,0,1};
    const char *modelPath = "MLModels/Generator.onnx";

    auto envLocal = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_INFO, "ENV");
    fEnv = std::move(envLocal);

    // Creating a OrtApi Class variable for getting access to C api, necessary for CUDA and TensorRT EP.
    const auto &ortApi = Ort::GetApi();
    //fSessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    // std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    // for(std::string ep : availableProviders ){
    //   G4cout << ep << G4endl;
    // }
    //  save json file for model execution profiling


    ///######## DNNL##############
    bool enable_cpu_mem_arena = true;

    // Currently, DNNL EP is not shown in the docs
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(fSessionOptions, enable_cpu_mem_arena));

    /*
    ///######## OpenVINO##############
    OrtOpenVINOProviderOptions ov_options;
    ov_options.device_type = "CPU_FP32";
    // ov_options.enable_vpu_fast_compile = 0;
    // ov_options.device_id = "";
    ov_options.num_of_threads = 1;
    // ov_options.use_compiled_network = false;
    // ov_options.blob_dump_path = "";
    // ov_options.context = "0x123456ff";  // For OpenCL, needs OpenVINO EP to be build with OpenCL flags
    // ov_options.enable_opencl_throttling = false;

    // fSessionOptions.AppendExecutionProvider_OpenVINO(ov_options);
    Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_OpenVINO(fSessionOptions, &ov_options));
    */

    auto sessionLocal = std::make_unique<Ort::Session>(*fEnv, modelPath, fSessionOptions);
    fSession = std::move(sessionLocal);
    std::cout << "Inference Session created" << std::endl;
    fInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);


    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_node_dims;
    size_t num_input_nodes = fSession->GetInputCount();
    std::vector<const char *> input_node_names(num_input_nodes);
    for (std::size_t i = 0; i < num_input_nodes; i++)
    {
        char *input_name = fSession->GetInputName(i, allocator);
        fInames = {input_name};
        input_node_names[i] = input_name;
        Ort::TypeInfo type_info = fSession->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        input_node_dims = tensor_info.GetShape();
        for (int j = 0; j < input_node_dims.size(); j++)
        {
        if (input_node_dims[j] < 0)
            input_node_dims[j] = 1;
        }
    }
    // output nodes
    std::vector<int64_t> output_node_dims;
    size_t num_output_nodes = fSession->GetOutputCount();
    std::vector<const char *> output_node_names(num_output_nodes);
    for (std::size_t i = 0; i < num_output_nodes; i++)
    {
        char *output_name = fSession->GetOutputName(i, allocator);
        output_node_names[i] = output_name;
        Ort::TypeInfo type_info = fSession->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        output_node_dims = tensor_info.GetShape();
        for (int j = 0; j < output_node_dims.size(); j++)
        {
        if (output_node_dims[j] < 0)
            output_node_dims[j] = 1;
        }
    }
    float genVector[(unsigned)(aGenVector.size())];
    for (int i = 0; i < (unsigned)(aGenVector.size()); i++)
        genVector[i] = aGenVector[i];
    int values_length = sizeof(genVector) / sizeof(genVector[0]);
    std::vector<int64_t> dims = {1, (unsigned)(aGenVector.size())};
    Ort::Value Input_noise_tensor =
        Ort::Value::CreateTensor<float>(fInfo, genVector, values_length, dims.data(), dims.size());
    assert(Input_noise_tensor.IsTensor());
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(Input_noise_tensor));
    // run the inference session
    std::cout << "Running Inference" << std::endl;
    std::vector<Ort::Value> ort_outputs =
        fSession->Run(Ort::RunOptions{nullptr}, fInames.data(), ort_inputs.data(), ort_inputs.size(),
                        output_node_names.data(), output_node_names.size());
    std::cout << "Successfully infered" << std::endl;
    // get pointer to output tensor float values
    float *floatarr = ort_outputs.front().GetTensorMutableData<float>();
    std::cout << "Got output reference" << std::endl;
    //aEnergies.assign(aSize, 0);
    //for (int i = 0; i < aSize; ++i)
    //    aEnergies[i] = floatarr[i];

    std::cout << "Inference Complete" << std::endl;

    //Ort::OrtRelease(fSessionOptions);
    //Ort::OrtRelease(fSession);
    //Ort::OrtRelease(fInfo);
    //Ort::OrtRelease(fEnv);


    return 0;
}

// Compile Command
/*g++ -L/home/priyammehta/onnxruntime/install/lib -L/home/priyammehta/onednn/install/lib -L/home/priyammehta/intel/openvino_2022/runtime/lib \
        -I/home/priyammehta/onnxruntime/install/include/onnxruntime/core/session -I/home/priyammehta/onnxruntime/install/include/onnxruntime/core/providers/openvino \
        -I/home/priyammehta/onnxruntime/install/include/onnxruntime/core/providers/dnnl \
        -I/home/priyammehta/onednn/install/include -I/home/priyammehta/intel/openvino_2022/runtime/include \
        ORTsample.cc

g++ -L/home/priyammehta/onnxruntime/install/lib \
    -L/home/priyammehta/onednn/install/lib \
    -L/home/priyammehta/intel/openvino_2022/runtime/lib \
    -I/home/priyammehta/onnxruntime/install/include/onnxruntime/core/session \
    -I/home/priyammehta/onnxruntime/install/include/onnxruntime/core \
    -I/home/priyammehta/onednn/install/include \
    -I/home/priyammehta/intel/openvino_2022/runtime/include \
    ORTsample.cc \
    -lonnxruntime \
    -lonnxruntime_providers_dnnl \
    -lonnxruntime_providers_openvino \
    -lonnxruntime_providers_shared
*/