//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//

// Notes for me while I am updating the file to include Dnnl, OpenVino, Cuda and Tensorrt EPs
// OpenVino - Disable all ORT Graph Optimizations, OpenVino does its own and it is observed that
//            giving all control to OpenVino performs the best.
// https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#onnxruntime-graph-optimization-level
//
// Dnnl - None so far
//
// Cuda - None so far
//
// Tensorrt - None so far

#ifdef USE_INFERENCE_ONNX
#include "Par04InferenceInterface.hh"
#include "G4RotationMatrix.hh"
#include "Par04OnnxInference.hh"
#include "Par04OnnxExecutionProviders.hh"
#include <vector>
#include <cassert>
#include <variant>
#ifdef USE_ROOT
#include "TSystem.h"
#endif
#ifdef USE_CUDA
//#include "cuda.h"
#include "cuda_runtime_api.h"
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04OnnxInference::Par04OnnxInference(G4String modelPath, G4int profileFlag, G4int optimizeFlag, G4int intraOpNumThreads,
                                      G4int dnnlFlag, G4int openvinoFlag, G4int cudaFlag, G4int tensorrtFlag,
                                      G4bool fDnnlEnableCpuMemArena,
                                      std::vector<std::variant<const char *, int>> &openvino_options,
                                      std::vector<const char *> &cuda_keys,
                                      std::vector<const char *> &cuda_values,     
                                      std::vector<const char *> &trt_keys,     
                                      std::vector<const char *> &trt_values)
    : Par04InferenceInterface()
{
  // initialization of the enviroment and inference session
  auto envLocal = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "ENV");
  fEnv = std::move(envLocal);
  //fEnv = new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "ENV");
  // Creating a OrtApi Class variable for getting access to C api, necessary for CUDA and TensorRT EP.
  const auto &ortApi = Ort::GetApi();

  // Alternative way
  // auto ortApibase = OrtGetApiBase();
  // auto ortApi = ortApibase->GetApi(ORT_API_VERSION);

  // graph optimizations of the model
  // if the flag is not set to true none of the optimizations will be applied
  // if it is set to true all the optimizations will be applied
  if (optimizeFlag)
  {
    fSessionOptions.SetOptimizedModelFilePath("opt-graph");
    fSessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    G4cout << "Optimization Enabled! If you get an error regarding \
               compiled nodes, turn off optimization and try again" << G4endl;
    // ORT_ENABLE_BASIC #### ORT_ENABLE_EXTENDED
  }
  else
    fSessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

  #ifdef USE_DNNL
  if (dnnlFlag)
  {
    fSessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(fSessionOptions, fDnnlEnableCpuMemArena));
    G4cout << "Added oneDNN Execution Provider" << G4endl;
  }
  #endif
  #ifdef USE_OPENVINO
  if (openvinoFlag)
  {
    OrtOpenVINOProviderOptions ov_options;
    ov_options.device_type = std::get<const char *> (openvino_options[0]);
    ov_options.enable_vpu_fast_compile = std::get<int> (openvino_options[1]);   
    ov_options.device_id = std::get<const char *> (openvino_options[2]);                 
    ov_options.num_of_threads = std::get<int> (openvino_options[3]);
    ov_options.use_compiled_network = std::get<int> (openvino_options[4]);       
    ov_options.blob_dump_path = std::get<const char *> (openvino_options[5]);                             

    fSessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    fSessionOptions.AppendExecutionProvider_OpenVINO(ov_options);
    fSessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    
    G4cout << "Added OpenVINO Execution Provider" << G4endl;
  }
  #endif
  #ifdef USE_TENSORRT
  if (tensorrtFlag)
  {
    OrtTensorRTProviderOptionsV2 *fTrtOptions = nullptr;
    Ort::ThrowOnError(ortApi.CreateTensorRTProviderOptions(&fTrtOptions));
    /*std::vector<const char *> trt_keys{
        "device_id",
        "trt_max_workspace_size",
        "trt_max_partition_iterations",
        "trt_min_subgraph_size",
        "trt_fp16_enable",
        "trt_int8_enable",
        "trt_int8_use_native_calibration_table",
        "trt_engine_cache_enable",
        "trt_engine_cache_path",
        "trt_dump_subgraphs",
    };
    std::vector<const char *> trt_values{
        "0",                     // device_id
        "2147483648",            // trt_max_workspace_size
        "10",                    // trt_max_partition_iterations
        "5",                     // trt_min_subgraph_size
        "0",                     // trt_fp16_enable
        "0",                     // trt_int8_enable
        "1",                     // trt_int8_use_native_calibration_table
        "1",                     // trt_engine_cache_enable
        "/opt/trt/geant4/cache", // trt_engine_cache_path
        "1"                      // trt_dump_subgraphs
    };
    */
    Ort::ThrowOnError(ortApi.UpdateTensorRTProviderOptions(fTrtOptions, trt_keys.data(), trt_values.data(), trt_keys.size()));
    Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_TensorRT_V2(fSessionOptions, fTrtOptions));
    G4cout << "Added TensorRT Execution Provider" << G4endl;
  }
  #endif
  #ifdef USE_CUDA
  if (cudaFlag)
  {
    OrtCUDAProviderOptionsV2 *fCudaOptions = nullptr;
    Ort::ThrowOnError(ortApi.CreateCUDAProviderOptions(&fCudaOptions));
    /*
    std::vector<const char *> cuda_keys{
        "device_id",
        "gpu_mem_limit",
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
    };
    std::vector<const char *> cuda_values{
        "0",                // device_id
        "2147483648",       // gpu_mem_limit
        "kSameAsRequested", // arena_extend_strategy
        "DEFAULT",          // cudnn_conv_algo_search
        "1",                // do_copy_in_default_stream
        "1",                // cudnn_conv_use_max_workspace
    };
    */
    Ort::ThrowOnError(ortApi.UpdateCUDAProviderOptions(fCudaOptions, cuda_keys.data(), cuda_values.data(), cuda_keys.size()));
    Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_CUDA_V2(fSessionOptions, fCudaOptions));
    G4cout << "Added CUDA Execution Provider" << G4endl;
  }
  #endif
  if (profileFlag)
    fSessionOptions.EnableProfiling("opt.json");

  auto sessionLocal = std::make_unique<Ort::Session>(*fEnv, modelPath, fSessionOptions);
  fSession = std::move(sessionLocal);
  //fSession = new Ort::Session(*fEnv, modelPath, fSessionOptions);
  G4cout << "Inference Session created" << G4endl;
  fInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04OnnxInference::~Par04OnnxInference(){
  //fSession->release();
  //fEnv->release();
  G4cout << "Onnx Inference destroyed!" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04OnnxInference::RunInference(vector<float> aGenVector, std::vector<G4double> &aEnergies,
                                      int aSize)
//)
{
  /*
  G4cout << "\nInside Run" << G4endl;
  const G4float toMB = 1.f / 1024.f;
  // input nodes
  #ifdef USE_CUDA
  size_t free, total;
  G4int num_gpus;
  G4double tot_res_mem_before = 0;
	G4double tot_virt_mem_before = 0;
  cudaGetDeviceCount( &num_gpus );
  G4double usage_mem_gpu_before = 0;
  for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        //int id;
        //cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        usage_mem_gpu_before += total - free;
        //std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
  }
  G4cout << "GPU mem before Run: " << usage_mem_gpu_before * toMB << G4endl;
  #endif
  #ifdef USE_ROOT
  static ProcInfo_t info;
  gSystem->GetProcInfo(&info);
  tot_res_mem_before = info.fMemResident * toMB;
	tot_virt_mem_before = info.fMemVirtual * toMB;
  G4cout << "CPU resident mem before Run: " << tot_res_mem_before << "\n"
         << "CPU virtual mem before Run: " << tot_virt_mem_before << G4endl;
  #endif
  */
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
  G4cout << "Got input and output nodes" << G4endl;
  // create input tensor object from data values
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
  G4cout << "Running Inference" << G4endl;
  std::vector<Ort::Value> ort_outputs =
      fSession->Run(Ort::RunOptions{nullptr}, fInames.data(), ort_inputs.data(), ort_inputs.size(),
                   output_node_names.data(), output_node_names.size());
  G4cout << "Successfully infered" << G4endl;
  // get pointer to output tensor float values
  float *floatarr = ort_outputs.front().GetTensorMutableData<float>();
  G4cout << "Got output reference" << G4endl;
  aEnergies.assign(aSize, 0);
  for (int i = 0; i < aSize; ++i)
    aEnergies[i] = floatarr[i];

  G4cout << "Inference Complete" << G4endl;
// Memory Usage extraction
  /*
  #ifdef USE_ROOT
  gSystem->GetProcInfo(&info);
  G4double tot_res_mem_after = info.fMemResident * toMB;
	G4double tot_virt_mem_after = info.fMemVirtual * toMB;
  G4double tot_res_mem = tot_res_mem_after - tot_res_mem_before;
	G4double tot_virt_mem = tot_virt_mem_after - tot_virt_mem_before;
  //analysisManager->FillNtupleDColumn(6, tot_res_mem);
  //analysisManager->FillNtupleDColumn(7, tot_virt_mem);
  G4cout << "CPU resident mem after Run: " << tot_res_mem_after << "\n"
         << "CPU virtual mem after Run: " << tot_virt_mem_after << G4endl;
  G4cout << "CPU resident mem usage: " << tot_res_mem << "\n"
         << "CPU virtual mem usage: " << tot_virt_mem << G4endl;

  //tot_res_mem_before = 0;
  //tot_virt_mem_before = 0;
  #endif
  #ifdef USE_CUDA
  size_t usage_mem_gpu_after = 0;
  cudaGetDeviceCount( &num_gpus );
  for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        usage_mem_gpu_after += total - free;
        //std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
  }
  G4cout << "GPU mem after Run: " << usage_mem_gpu_after * toMB << G4endl;
  size_t usage_mem_gpu = usage_mem_gpu_after - usage_mem_gpu_before;
  G4cout << "GPU mem usage: " << usage_mem_gpu * toMB << G4endl;
  //analysisManager->FillNtupleDColumn(8, usage_mem_gpu * toMB);
  
  //usage_mem_gpu_before = 0;
  #endif
  */
}

#endif
