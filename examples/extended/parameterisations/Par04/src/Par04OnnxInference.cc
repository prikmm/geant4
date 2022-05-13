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
#include <cassert>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04OnnxInference::Par04OnnxInference(G4String modelPath, G4int profileFlag, G4int optimizeFlag,
                                       G4int intraOpNumThreads, G4bool fDnnlEpFlag, G4bool fOpenVinoEpFlag,
                                       G4bool fCudaEpFlag, G4bool fTensorrtEpFlag)
  : Par04InferenceInterface()
{
  // initialization of the enviroment and inference session
  auto envLocal = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ENV");
  fEnv          = std::move(envLocal);
  // graph optimizations of the model
  // if the flag is not set to true none of the optimizations will be applied
  // if it is set to true all the optimizations will be applied
  if(fDnnlEpFlag)
  {
    fSessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    if(optimizeFlag)
    {
      fSessionOptions.SetOptimizedModelFilePath("opt-graph");
      fSessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
      // ORT_ENABLE_BASIC #### ORT_ENABLE_EXTENDED
    }
    else
      fSessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    // save json file for model execution profiling
    bool enable_cpu_mem_arena = true;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(fSessionOptions, enable_cpu_mem_arena));
  }
  else if(fOpenVinoEpFlag)
  {
    OrtOpenVINOProviderOptions ov_options;
    ov_options.device_type = "CPU_FP32";
    ov_options.enable_vpu_fast_compile = 0;
    ov_options.device_id = "";
    ov_options.num_of_threads = intraOpNumThreads;
    ov_options.use_compiled_network = false;
    ov_options.blob_dump_path = "";
    ov_options.context = 0x123456ff;
    ov_options.enable_opencl_throttling = false;
    Ort::ThrowOnError(SessionOptionsAppendExecutionProvider_OpenVINO(fSessionOptions, &options));
    fSessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

  }
  else if(fCudaEpFlag)
  {

  }
  else if(fTensorrtEpFlag)
  {
    int device_id = 0;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(fSessionOptions, device_id));
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(fSessionOptions, device_id));
  }
  if(profileFlag)
    fSessionOptions.EnableProfiling("opt.json");


  auto sessionLocal = std::make_unique<Ort::Session>(*fEnv, modelPath, fSessionOptions);
  fSession          = std::move(sessionLocal);
  fInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04OnnxInference::RunInference(vector<float> aGenVector, std::vector<G4double>& aEnergies,
                                      int aSize)
{
  // input nodes
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<int64_t> input_node_dims;
  size_t num_input_nodes = fSession->GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  for(std::size_t i = 0; i < num_input_nodes; i++)
  {
    char* input_name               = fSession->GetInputName(i, allocator);
    fInames                        = { input_name };
    input_node_names[i]            = input_name;
    Ort::TypeInfo type_info        = fSession->GetInputTypeInfo(i);
    auto tensor_info               = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    input_node_dims                = tensor_info.GetShape();
    for(int j = 0; j < input_node_dims.size(); j++)
    {
      if(input_node_dims[j] < 0)
        input_node_dims[j] = 1;
    }
  }
  // output nodes
  std::vector<int64_t> output_node_dims;
  size_t num_output_nodes = fSession->GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);
  for(std::size_t i = 0; i < num_output_nodes; i++)
  {
    char* output_name              = fSession->GetOutputName(i, allocator);
    output_node_names[i]           = output_name;
    Ort::TypeInfo type_info        = fSession->GetOutputTypeInfo(i);
    auto tensor_info               = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    output_node_dims               = tensor_info.GetShape();
    for(int j = 0; j < output_node_dims.size(); j++)
    {
      if(output_node_dims[j] < 0)
        output_node_dims[j] = 1;
    }
  }

  // create input tensor object from data values
  float genVector[(unsigned) (aGenVector.size())];
  for(int i = 0; i < (unsigned) (aGenVector.size()); i++)
    genVector[i] = aGenVector[i];
  int values_length         = sizeof(genVector) / sizeof(genVector[0]);
  std::vector<int64_t> dims = { 1, (unsigned) (aGenVector.size()) };
  Ort::Value Input_noise_tensor =
    Ort::Value::CreateTensor<float>(fInfo, genVector, values_length, dims.data(), dims.size());
  assert(Input_noise_tensor.IsTensor());
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(Input_noise_tensor));
  // run the inference session
  std::vector<Ort::Value> ort_outputs =
    fSession->Run(Ort::RunOptions{ nullptr }, fInames.data(), ort_inputs.data(), ort_inputs.size(),
                  output_node_names.data(), output_node_names.size());
  // get pointer to output tensor float values
  float* floatarr = ort_outputs.front().GetTensorMutableData<float>();
  aEnergies.assign(aSize, 0);
  for(int i = 0; i < aSize; ++i)
    aEnergies[i] = floatarr[i];
  
}

#endif
