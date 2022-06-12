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
#ifdef USE_INFERENCE
#include "Par04InferenceMessenger.hh"
#include "Par04InferenceSetup.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04InferenceMessenger::Par04InferenceMessenger(Par04InferenceSetup* aInference)
  : G4UImessenger()
  , fInference(aInference)
{
  fExampleDir = new G4UIdirectory("/Par04/");
  fExampleDir->SetGuidance("UI commands specific to this example");

  fInferenceDir = new G4UIdirectory("/Par04/inference/");
  fInferenceDir->SetGuidance("Inference construction UI commands");

  fInferenceLibraryCmd = new G4UIcmdWithAString("/Par04/inference/setInferenceLibrary", this);
  fInferenceLibraryCmd->SetGuidance("Inference library.");
  fInferenceLibraryCmd->SetParameterName("InferenceLibrary", false);
  fInferenceLibraryCmd->AvailableForStates(G4State_Idle);
  fInferenceLibraryCmd->SetToBeBroadcasted(true);

  fSizeLatentVectorCmd = new G4UIcmdWithAnInteger("/Par04/inference/setSizeLatentVector", this);
  fSizeLatentVectorCmd->SetGuidance("Set size of the latent space vector.");
  fSizeLatentVectorCmd->SetParameterName("SizeLatentVector", false);
  fSizeLatentVectorCmd->SetRange("SizeLatentVector>0");
  fSizeLatentVectorCmd->AvailableForStates(G4State_Idle);
  fSizeLatentVectorCmd->SetToBeBroadcasted(true);

  fSizeConditionVectorCmd =
    new G4UIcmdWithAnInteger("/Par04/inference/setSizeConditionVector", this);
  fSizeConditionVectorCmd->SetGuidance("Set size of the condition vector.");
  fSizeConditionVectorCmd->SetParameterName("SizeConditionVector", false);
  fSizeConditionVectorCmd->SetRange("SizeConditionVector>0");
  fSizeConditionVectorCmd->AvailableForStates(G4State_Idle);
  fSizeConditionVectorCmd->SetToBeBroadcasted(true);

  fModelPathNameCmd = new G4UIcmdWithAString("/Par04/inference/setModelPathName", this);
  fModelPathNameCmd->SetGuidance("Model path and name.");
  fModelPathNameCmd->SetParameterName("Name", false);
  fModelPathNameCmd->AvailableForStates(G4State_Idle);
  fModelPathNameCmd->SetToBeBroadcasted(true);

  fProfileFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setProfileFlag", this);
  fProfileFlagCmd->SetGuidance("Flag to save a json file for model execution profiling.");
  fProfileFlagCmd->SetParameterName("ProfileFlag", false);
  fProfileFlagCmd->SetRange("ProfileFlag>-1");
  fProfileFlagCmd->AvailableForStates(G4State_Idle);
  fProfileFlagCmd->SetToBeBroadcasted(true);

  fOptimizationFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setOptimizationFlag", this);
  fOptimizationFlagCmd->SetGuidance("Set optimization flag");
  fOptimizationFlagCmd->SetParameterName("OptimizationFlag", false);
  fOptimizationFlagCmd->SetRange("OptimizationFlag>-1");
  fOptimizationFlagCmd->AvailableForStates(G4State_Idle);
  fOptimizationFlagCmd->SetToBeBroadcasted(true);

  // Onnx Runtime Execution Provider flag commands
  fDnnlFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setDnnlFlag", this);
  fDnnlFlagCmd->SetGuidance("Whether to use DNNL Execution Provider for Onnx Runtime or not");
  fDnnlFlagCmd->SetParameterName("DnnlFlag", false);
  fDnnlFlagCmd->SetRange("DnnlFlag>-1");
  fDnnlFlagCmd->AvailableForStates(G4State_Idle);
  fDnnlFlagCmd->SetToBeBroadcasted(true);

  fOpenVinoFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setOpenVinoFlag", this);
  fOpenVinoFlagCmd->SetGuidance("Whether to use OpenVino Execution Provider for Onnx Runtime or not");
  fOpenVinoFlagCmd->SetParameterName("OpenVinoFlag", false);
  fOpenVinoFlagCmd->SetRange("OpenVinoFlag>-1");
  fOpenVinoFlagCmd->AvailableForStates(G4State_Idle);
  fOpenVinoFlagCmd->SetToBeBroadcasted(true);

  fCudaFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setCudaFlag", this);
  fCudaFlagCmd->SetGuidance("Whether to use CUDA Execution Provider for Onnx Runtime or not");
  fCudaFlagCmd->SetParameterName("CudaFlag", false);
  fCudaFlagCmd->SetRange("CudaFlag>-1");
  fCudaFlagCmd->AvailableForStates(G4State_Idle);
  fCudaFlagCmd->SetToBeBroadcasted(true);

  fTensorrtFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setTensorrtFlag", this);
  fTensorrtFlagCmd->SetGuidance("Whether to use TensorRT Execution Provider for Onnx Runtime or not");
  fTensorrtFlagCmd->SetParameterName("TensorrtFlag", false);
  fTensorrtFlagCmd->SetRange("TensorrtFlag>-1");
  fTensorrtFlagCmd->AvailableForStates(G4State_Idle);
  fTensorrtFlagCmd->SetToBeBroadcasted(true);

  /// OnnxRuntime Execution Provider Options
  /// oneDnn
  fDnnlOptionsDir = new G4UIdirectory("/Par04/inference/onednn");
  fDnnlOptionsDir->SetGuidance("Commands for setting options for oneDNN execution provider");

  fDnnlEnableCpuMemArenaCmd = new G4UIcmdWithABool("/Par04/inference/onednn/setEnableCpuMemArena", this);
  fDnnlEnableCpuMemArenaCmd->SetGuidance("Enable use of CPU Memory Arena when using oneDNN/DNNL Execution Provider");
  fDnnlEnableCpuMemArenaCmd->SetParameterName("DnnlEnableCpuMemArena", false);
  fDnnlEnableCpuMemArenaCmd->AvailableForStates(G4State_Idle);
  fDnnlEnableCpuMemArenaCmd->SetToBeBroadcasted(true);

  /// Cuda
  fCudaOptionsDir = new G4UIdirectory("/Par04/inference/cuda");
  fCudaOptionsDir->SetGuidance("Commands for setting options for Cuda execution provider");

  fCudaDeviceIdCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setDeviceId", this);
  fCudaDeviceIdCmd->SetGuidance("Device ID of Device on which to run CUDA code");
  fCudaDeviceIdCmd->SetParameterName("CudaDeviceId", false);
  fCudaDeviceIdCmd->AvailableForStates(G4State_Idle);
  fCudaDeviceIdCmd->SetToBeBroadcasted(true);

  fCudaGpuMemLimitCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setGpuMemLimit", this);
  fCudaGpuMemLimitCmd->SetGuidance("GPU Memory limit for CUDA");
  fCudaGpuMemLimitCmd->SetParameterName("CudaGpuMemLimit", false);
  fCudaGpuMemLimitCmd->AvailableForStates(G4State_Idle);
  fCudaGpuMemLimitCmd->SetToBeBroadcasted(true);

  fCudaArenaExtendedStrategyCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setArenaExtendedStrategy", this);
  fCudaArenaExtendedStrategyCmd->SetGuidance("Strategy for extending the device memory arena for CUDA");
  fCudaArenaExtendedStrategyCmd->SetParameterName("CudaArenaExtendedStrategy", false);
  fCudaArenaExtendedStrategyCmd->AvailableForStates(G4State_Idle);
  fCudaArenaExtendedStrategyCmd->SetToBeBroadcasted(true);

  fCudaCudnnConvAlgoSearchCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setCudnnConvAlgoSearch", this);
  fCudaCudnnConvAlgoSearchCmd->SetGuidance("Set which cuDNN Convolution Operation to use");
  fCudaCudnnConvAlgoSearchCmd->SetParameterName("CudaCudnnConvAlgoSearch", false);
  fCudaCudnnConvAlgoSearchCmd->AvailableForStates(G4State_Idle);
  fCudaCudnnConvAlgoSearchCmd->SetToBeBroadcasted(true);

  fCudaDoCopyInDefaultStreamCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setDoCopyInDefaultStream", this);
  fCudaDoCopyInDefaultStreamCmd->SetGuidance("Whether to use same stream for copying");
  fCudaDoCopyInDefaultStreamCmd->SetParameterName("CudaDoCopyInDefaultStream", false);
  fCudaDoCopyInDefaultStreamCmd->AvailableForStates(G4State_Idle);
  fCudaDoCopyInDefaultStreamCmd->SetToBeBroadcasted(true);

  fCudaCudnnConvUseMaxWorkspaceCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setCudnnConvUseMaxWorkspace", this);
  fCudaCudnnConvUseMaxWorkspaceCmd->SetGuidance("Memory Limit for cuDNN convolution operations");
  fCudaCudnnConvUseMaxWorkspaceCmd->SetParameterName("CudaCudnnConvUseMaxWorkspace", false);
  fCudaCudnnConvUseMaxWorkspaceCmd->AvailableForStates(G4State_Idle);
  fCudaCudnnConvUseMaxWorkspaceCmd->SetToBeBroadcasted(true);

  /// TensorRT
  fTensorRTOptionsDir = new G4UIdirectory("/Par04/inference/trt");
  fTensorRTOptionsDir->SetGuidance("Commands for setting options for TensorRT execution provider");

  fTrtDeviceIdCmd = new G4UIcmdWithAString("/Par04/inference/trt/setDeviceId", this);
  fTrtDeviceIdCmd->SetGuidance("Device ID of the Device on which to run TensorRT code");
  fTrtDeviceIdCmd->SetParameterName("TrtDeviceId", false);
  fTrtDeviceIdCmd->AvailableForStates(G4State_Idle);
  fTrtDeviceIdCmd->SetToBeBroadcasted(true);

  fTrtMaxWorkspaceSizeCmd = new G4UIcmdWithAString("/Par04/inference/trt/setMaxWorkspaceSize", this);
  fTrtMaxWorkspaceSizeCmd->SetGuidance("Memory Size to use for TensorRT");
  fTrtMaxWorkspaceSizeCmd->SetParameterName("TrtMaxWorkspaceSize", false);
  fTrtMaxWorkspaceSizeCmd->AvailableForStates(G4State_Idle);
  fTrtMaxWorkspaceSizeCmd->SetToBeBroadcasted(true);

  fTrtMaxPartitionIterationsCmd = new G4UIcmdWithAString("/Par04/inference/trt/setMaxPartitionIterations", this);
  fTrtMaxPartitionIterationsCmd->SetGuidance("Maximum Iterations allowed for model partitioning in TensorRT");
  fTrtMaxPartitionIterationsCmd->SetParameterName("TrtMaxPartitionIterations", false);
  fTrtMaxPartitionIterationsCmd->AvailableForStates(G4State_Idle);
  fTrtMaxPartitionIterationsCmd->SetToBeBroadcasted(true);

  fTrtMinSubgraphSizeCmd = new G4UIcmdWithAString("/Par04/inference/trt/setMinSubgraphSize", this);
  fTrtMinSubgraphSizeCmd->SetGuidance("Minimum node size in a subgraph after partitioning");
  fTrtMinSubgraphSizeCmd->SetParameterName("TrtMinSubgraphSize", false);
  fTrtMinSubgraphSizeCmd->AvailableForStates(G4State_Idle);
  fTrtMinSubgraphSizeCmd->SetToBeBroadcasted(true);

  fTrtFp16EnableCmd = new G4UIcmdWithAString("/Par04/inference/trt/setFp16Enable", this);
  fTrtFp16EnableCmd->SetGuidance("Whether to use 16 bit Floating Point computation");
  fTrtFp16EnableCmd->SetParameterName("TrtFp16Enable", false);
  fTrtFp16EnableCmd->AvailableForStates(G4State_Idle);
  fTrtFp16EnableCmd->SetToBeBroadcasted(true);

  fTrtInt8EnableCmd = new G4UIcmdWithAString("/Par04/inference/trt/setInt8Enable", this);
  fTrtInt8EnableCmd->SetGuidance("Whether to use 8bit Integer computation");
  fTrtInt8EnableCmd->SetParameterName("TrtInt8Enable", false);
  fTrtInt8EnableCmd->AvailableForStates(G4State_Idle);
  fTrtInt8EnableCmd->SetToBeBroadcasted(true);

  fTrtInt8UseNativeCalibrationTableCmd = new G4UIcmdWithAString("/Par04/inference/trt/etInt8UseNativeCalibrationTable", this);
  fTrtInt8UseNativeCalibrationTableCmd->SetGuidance("Select what calibration table is used for non-QDQ models in INT8 mode. If 1, native TensorRT generated calibration table is used; if 0, ONNXRUNTIME tool generated calibration table is used.");
  fTrtInt8UseNativeCalibrationTableCmd->SetParameterName("TrtInt8UseNativeCalibrationTable", false);
  fTrtInt8UseNativeCalibrationTableCmd->AvailableForStates(G4State_Idle);
  fTrtInt8UseNativeCalibrationTableCmd->SetToBeBroadcasted(true);

  fTrtEngineCacheEnableCmd = new G4UIcmdWithAString("/Par04/inference/trt/setEngineCacheEnable", this);
  fTrtEngineCacheEnableCmd->SetGuidance("Enable TensorRT engine caching. The purpose of using engine caching is to save engine build time in the case that TensorRT may take long time to optimize and build engine.");
  fTrtEngineCacheEnableCmd->SetParameterName("TrtEngineCacheEnable", false);
  fTrtEngineCacheEnableCmd->AvailableForStates(G4State_Idle);
  fTrtEngineCacheEnableCmd->SetToBeBroadcasted(true);

  fTrtEngineCachePathCmd = new G4UIcmdWithAString("/Par04/inference/trt/setEngineCachePath", this);
  fTrtEngineCachePathCmd->SetGuidance("Specify path for TensorRT engine and profile files");
  fTrtEngineCachePathCmd->SetParameterName("TrtEngineCachePath", false);
  fTrtEngineCachePathCmd->AvailableForStates(G4State_Idle);
  fTrtEngineCachePathCmd->SetToBeBroadcasted(true);

  fTrtDumpSubgraphsCmd = new G4UIcmdWithAString("/Par04/inference/trt/setDumpSubgraphs", this);
  fTrtDumpSubgraphsCmd->SetGuidance("Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem.");
  fTrtDumpSubgraphsCmd->SetParameterName("TrtDumpSubgraphs", false);
  fTrtDumpSubgraphsCmd->AvailableForStates(G4State_Idle);
  fTrtDumpSubgraphsCmd->SetToBeBroadcasted(true);

  /// OpenVINO
  fOpenVINOOptionsDir = new G4UIdirectory("/Par04/inference/openvino");
  fOpenVINOOptionsDir->SetGuidance("Commands for setting options for OpenVINO execution provider");

  fOVDeviceTypeCmd = new G4UIcmdWithAString("/Par04/inference/openvino/setDeviceType", this);
  fOVDeviceTypeCmd->SetGuidance("Device Type on which to run OpenVINO");
  fOVDeviceTypeCmd->SetParameterName("OVDeviceType", false);
  fOVDeviceTypeCmd->AvailableForStates(G4State_Idle);
  fOVDeviceTypeCmd->SetToBeBroadcasted(true);

  fOVEnableVpuFastCompileCmd = new G4UIcmdWithAnInteger("/Par04/inference/openvino/setEnableVpuFastCompile", this);
  fOVEnableVpuFastCompileCmd->SetGuidance("During initialization of the VPU device with compiled model, Fast-compile may be optionally enabled to speeds up the model’s compilation to VPU device specific format");
  fOVEnableVpuFastCompileCmd->SetParameterName("OVEnableVpuFastCompile", false);
  fOVEnableVpuFastCompileCmd->AvailableForStates(G4State_Idle);
  fOVEnableVpuFastCompileCmd->SetToBeBroadcasted(true);

  fOVDeviceIdCmd = new G4UIcmdWithAString("/Par04/inference/openvino/setDeviceId", this);
  fOVDeviceIdCmd->SetGuidance("Device ID on which to run OpenVINO");
  fOVDeviceIdCmd->SetParameterName("OVDeviceId", false);
  fOVDeviceIdCmd->AvailableForStates(G4State_Idle);
  fOVDeviceIdCmd->SetToBeBroadcasted(true);

  fOVNumOfThreadsCmd = new G4UIcmdWithAnInteger("/Par04/inference/openvino/setNumOfThreads", this);
  fOVNumOfThreadsCmd->SetGuidance("Number of threads to use for OpenVINO runtime");
  fOVNumOfThreadsCmd->SetParameterName("OVNumOfThreads", false);
  fOVNumOfThreadsCmd->AvailableForStates(G4State_Idle);
  fOVNumOfThreadsCmd->SetToBeBroadcasted(true);

  fOVUseCompiledNetworkCmd = new G4UIcmdWithAnInteger("/Par04/inference/openvino/setUseCompiledNetwork", this);
  fOVUseCompiledNetworkCmd->SetGuidance("It can be used to directly import pre-compiled blobs if exists or dump a pre-compiled blob at the executable path.");
  fOVUseCompiledNetworkCmd->SetParameterName("OVUseCompiledNetwork", false);
  fOVUseCompiledNetworkCmd->AvailableForStates(G4State_Idle);
  fOVUseCompiledNetworkCmd->SetToBeBroadcasted(true);

  fOVBlobDumpPathCmd = new G4UIcmdWithAString("/Par04/inference/openvino/setBlobDumpPath", this);
  fOVBlobDumpPathCmd->SetGuidance("Explicitly specify the path where you would like to dump and load the blobs for the save/load blob feature when use_compiled_network setting is enabled.");
  fOVBlobDumpPathCmd->SetParameterName("OVBlobDumpPath", false);
  fOVBlobDumpPathCmd->AvailableForStates(G4State_Idle);
  fOVBlobDumpPathCmd->SetToBeBroadcasted(true);

  fOVEnableOpenCLThrottlingCmd = new G4UIcmdWithAString("/Par04/inference/openvino/setEnableOpenCLThrottling", this);
  fOVEnableOpenCLThrottlingCmd->SetGuidance("This option enables OpenCL queue throttling for GPU devices (reduces CPU utilization when using GPU).");
  fOVEnableOpenCLThrottlingCmd->SetParameterName("OVEnableOpenCLThrottling", false);
  fOVEnableOpenCLThrottlingCmd->AvailableForStates(G4State_Idle);
  fOVEnableOpenCLThrottlingCmd->SetToBeBroadcasted(true);

  //...................................................

  fMeshNbRhoCellsCmd = new G4UIcmdWithAnInteger("/Par04/inference/setNbOfRhoCells", this);
  fMeshNbRhoCellsCmd->SetGuidance("Set number of rho cells in the cylindrical mesh readout.");
  fMeshNbRhoCellsCmd->SetParameterName("NbRhoCells", false);
  fMeshNbRhoCellsCmd->SetRange("NbRhoCells>0");
  fMeshNbRhoCellsCmd->AvailableForStates(G4State_Idle);
  fMeshNbRhoCellsCmd->SetToBeBroadcasted(true);

  fMeshNbPhiCellsCmd = new G4UIcmdWithAnInteger("/Par04/inference/setNbOfPhiCells", this);
  fMeshNbPhiCellsCmd->SetGuidance("Set number of phi cells in the cylindrical mesh readout.");
  fMeshNbPhiCellsCmd->SetParameterName("NbPhiCells", false);
  fMeshNbPhiCellsCmd->SetRange("NbPhiCells>0");
  fMeshNbPhiCellsCmd->AvailableForStates(G4State_Idle);
  fMeshNbPhiCellsCmd->SetToBeBroadcasted(true);

  fMeshNbZCellsCmd = new G4UIcmdWithAnInteger("/Par04/inference/setNbOfZCells", this);
  fMeshNbZCellsCmd->SetGuidance("Set number of z cells in the cylindrical mesh readout.");
  fMeshNbZCellsCmd->SetParameterName("NbZCells", false);
  fMeshNbZCellsCmd->SetRange("NbZCells>0");
  fMeshNbZCellsCmd->AvailableForStates(G4State_Idle);
  fMeshNbZCellsCmd->SetToBeBroadcasted(true);

  fMeshSizeRhoCellsCmd = new G4UIcmdWithADoubleAndUnit("/Par04/inference/setSizeOfRhoCells", this);
  fMeshSizeRhoCellsCmd->SetGuidance("Set size of rho cells in the cylindrical readout mesh");
  fMeshSizeRhoCellsCmd->SetParameterName("Size", false);
  fMeshSizeRhoCellsCmd->SetRange("Size>0.");
  fMeshSizeRhoCellsCmd->SetUnitCategory("Length");
  fMeshSizeRhoCellsCmd->AvailableForStates(G4State_Idle);
  fMeshSizeRhoCellsCmd->SetToBeBroadcasted(true);

  fMeshSizeZCellsCmd = new G4UIcmdWithADoubleAndUnit("/Par04/inference/setSizeOfZCells", this);
  fMeshSizeZCellsCmd->SetGuidance("Set size of z cells in the cylindrical readout mesh");
  fMeshSizeZCellsCmd->SetParameterName("Size", false);
  fMeshSizeZCellsCmd->SetRange("Size>0.");
  fMeshSizeZCellsCmd->SetUnitCategory("Length");
  fMeshSizeZCellsCmd->AvailableForStates(G4State_Idle);
  fMeshSizeZCellsCmd->SetToBeBroadcasted(true);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04InferenceMessenger::~Par04InferenceMessenger()
{
  delete fInferenceLibraryCmd;
  delete fSizeLatentVectorCmd;
  delete fSizeConditionVectorCmd;
  delete fModelPathNameCmd;
  delete fProfileFlagCmd;
  delete fOptimizationFlagCmd;
  // Execution provider flags
  delete fDnnlFlagCmd;
  delete fOpenVinoFlagCmd;
  delete fCudaFlagCmd;
  delete fTensorrtFlagCmd;

  /// oneDNN commands
  delete fDnnlEnableCpuMemArenaCmd;

  /// Cuda Commands
  delete fCudaDeviceIdCmd;
  delete fCudaGpuMemLimitCmd;
  delete fCudaArenaExtendedStrategyCmd;
  delete fCudaCudnnConvAlgoSearchCmd;
  delete fCudaDoCopyInDefaultStreamCmd;
  delete fCudaCudnnConvUseMaxWorkspaceCmd;

  /// TensorRT Commands
  delete fTrtDeviceIdCmd;
  delete fTrtMaxWorkspaceSizeCmd;
  delete fTrtMaxPartitionIterationsCmd;
  delete fTrtMinSubgraphSizeCmd;
  delete fTrtFp16EnableCmd;
  delete fTrtInt8EnableCmd;
  delete fTrtInt8UseNativeCalibrationTableCmd;
  delete fTrtEngineCacheEnableCmd;
  delete fTrtEngineCachePathCmd;
  delete fTrtDumpSubgraphsCmd;

  /// OpenVINO commands
  delete fOVDeviceTypeCmd;
  delete fOVEnableVpuFastCompileCmd;
  delete fOVDeviceIdCmd;
  delete fOVNumOfThreadsCmd;
  delete fOVUseCompiledNetworkCmd;
  delete fOVBlobDumpPathCmd;
  delete fOVEnableOpenCLThrottlingCmd;

  //........................
  delete fMeshNbRhoCellsCmd;
  delete fMeshNbPhiCellsCmd;
  delete fMeshNbZCellsCmd;
  delete fMeshSizeRhoCellsCmd;
  delete fMeshSizeZCellsCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04InferenceMessenger::SetNewValue(G4UIcommand* aCommand, G4String aNewValue)
{
  if(aCommand == fInferenceLibraryCmd)
  {
    fInference->SetInferenceLibrary(aNewValue);
  }
  if(aCommand == fSizeLatentVectorCmd)
  {
    fInference->SetSizeLatentVector(std::stoi(aNewValue));
  }
  if(aCommand == fSizeConditionVectorCmd)
  {
    fInference->SetSizeConditionVector(std::stoi(aNewValue));
  }
  if(aCommand == fModelPathNameCmd)
  {
    fInference->SetModelPathName(aNewValue);
  }
  if(aCommand == fProfileFlagCmd)
  {
    fInference->SetProfileFlag(std::stoi(aNewValue));
  }
  if(aCommand == fOptimizationFlagCmd)
  {
    fInference->SetOptimizationFlag(std::stoi(aNewValue));
  }
  /// Onnx Runtime Execution Provider Flags
  if (aCommand == fDnnlFlagCmd)
  {
    fInference->SetDnnlFlag(std::stoi(aNewValue));
  } 
  if (aCommand == fOpenVinoFlagCmd)
  {
    fInference->SetOpenVinoFlag(std::stoi(aNewValue));
  }
  if (aCommand == fCudaFlagCmd)
  {
    fInference->SetCudaFlag(std::stoi(aNewValue));
  }
  if (aCommand == fTensorrtFlagCmd)
  {
    fInference->SetTensorrtFlag(std::stoi(aNewValue));
  } 
  ///// Execution Provider Options
  /// oneDNN
  if (aCommand == fDnnlEnableCpuMemArenaCmd)
  {
    fInference->SetDnnlEnableCpuMemArena(std::stoi(aNewValue));
  }

  /// Cuda
  if (aCommand == fCudaDeviceIdCmd)
  {
    fInference->SetCudaDeviceId(aNewValue);
  }
  else if (aCommand == fCudaGpuMemLimitCmd)
  {
    fInference->SetCudaGpuMemLimit(aNewValue);
  }
  else if (aCommand == fCudaArenaExtendedStrategyCmd)
  {
    fInference->SetCudaArenaExtendedStrategy(aNewValue);
  }
  else if (aCommand == fCudaCudnnConvAlgoSearchCmd)
  {
    fInference->SetCudaCudnnConvAlgoSearch(aNewValue);
  }
  else if (aCommand == fCudaDoCopyInDefaultStreamCmd)
  {
    fInference->SetCudaDoCopyInDefaultStream(aNewValue);
  }
  else if (aCommand == fCudaCudnnConvUseMaxWorkspaceCmd)
  {
    fInference->SetCudaCudnnConvUseMaxWorkspace(aNewValue);
  }

  /// TensorRT
  if (aCommand == fTrtDeviceIdCmd)
  {
    fInference->SetTrtDeviceId(aNewValue);
  }
  else if (aCommand == fTrtMaxWorkspaceSizeCmd)
  {
    fInference->SetTrtMaxWorkspaceSize(aNewValue);
  }
  else if (aCommand == fTrtMaxPartitionIterationsCmd)
  {
    fInference->SetTrtMaxPartitionIterations(aNewValue);
  }
  else if (aCommand == fTrtMinSubgraphSizeCmd)
  {
    fInference->SetTrtMinSubgraphSize(aNewValue);
  }
  else if (aCommand == fTrtFp16EnableCmd)
  {
    fInference->SetTrtFp16Enable(aNewValue);
  }
  else if (aCommand == fTrtInt8EnableCmd)
  {
    fInference->SetTrtInt8Enable(aNewValue);
  }
  else if (aCommand == fTrtInt8UseNativeCalibrationTableCmd)
  {
    fInference->SetTrtInt8UseNativeCalibrationTable(aNewValue);
  }
  else if (aCommand == fTrtEngineCacheEnableCmd)
  {
    fInference->SetTrtEngineCacheEnable(aNewValue);
  }
  else if (aCommand == fTrtEngineCachePathCmd)
  {
    fInference->SetTrtEngineCachePath(aNewValue);
  }
  else if (aCommand == fTrtDumpSubgraphsCmd)
  {
    fInference->SetTrtDumpSubgraphs(aNewValue);
  }

  /// OpenVINO
  if (aCommand == fOVDeviceTypeCmd)
  {
    fInference->SetOVDeviceType(aNewValue);
  }
  else if (aCommand == fOVEnableVpuFastCompileCmd)
  {
    fInference->SetOVEnableVpuFastCompile(std::stoi(aNewValue));
  }
  else if (aCommand == fOVDeviceIdCmd)
  {
    fInference->SetOVDeviceId(aNewValue);
  }
  else if (aCommand == fOVNumOfThreadsCmd)
  {
    fInference->SetOVNumOfThreads(std::stoi(aNewValue));
  }
  else if (aCommand == fOVUseCompiledNetworkCmd)
  {
    fInference->SetOVUseCompiledNetwork(std::stoi(aNewValue));
  }
  else if (aCommand == fOVBlobDumpPathCmd)
  {
    fInference->SetOVBlobDumpPath(aNewValue);
  }
  else if (aCommand == fOVEnableOpenCLThrottlingCmd)
  {
    fInference->SetOVEnableOpenCLThrottling(std::stoi(aNewValue));
  }

  /// ORT EP END
  if(aCommand == fMeshNbRhoCellsCmd)
  {
    fInference->SetMeshNbOfCells(0, fMeshNbRhoCellsCmd->GetNewIntValue(aNewValue));
  }
  else if(aCommand == fMeshNbPhiCellsCmd)
  {
    fInference->SetMeshNbOfCells(1, fMeshNbPhiCellsCmd->GetNewIntValue(aNewValue));
    fInference->SetMeshSizeOfCells(1,
                                   2. * CLHEP::pi / fMeshNbPhiCellsCmd->GetNewIntValue(aNewValue));
  }
  else if(aCommand == fMeshNbZCellsCmd)
  {
    fInference->SetMeshNbOfCells(2, fMeshNbZCellsCmd->GetNewIntValue(aNewValue));
  }
  else if(aCommand == fMeshSizeRhoCellsCmd)
  {
    fInference->SetMeshSizeOfCells(0, fMeshSizeRhoCellsCmd->GetNewDoubleValue(aNewValue));
  }
  else if(aCommand == fMeshSizeZCellsCmd)
  {
    fInference->SetMeshSizeOfCells(2, fMeshSizeZCellsCmd->GetNewDoubleValue(aNewValue));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4String Par04InferenceMessenger::GetCurrentValue(G4UIcommand* aCommand)
{
  G4String cv;

  if(aCommand == fInferenceLibraryCmd)
  {
    cv = fInferenceLibraryCmd->ConvertToString(fInference->GetInferenceLibrary());
  }
  if(aCommand == fSizeLatentVectorCmd)
  {
    cv = fSizeLatentVectorCmd->ConvertToString(fInference->GetSizeLatentVector());
  }
  if(aCommand == fSizeConditionVectorCmd)
  {
    cv = fSizeConditionVectorCmd->ConvertToString(fInference->GetSizeConditionVector());
  }
  if(aCommand == fModelPathNameCmd)
  {
    cv = fModelPathNameCmd->ConvertToString(fInference->GetModelPathName());
  }
  if(aCommand == fProfileFlagCmd)
  {
    cv = fProfileFlagCmd->ConvertToString(fInference->GetProfileFlag());
  }
  if(aCommand == fOptimizationFlagCmd)
  {
    cv = fOptimizationFlagCmd->ConvertToString(fInference->GetOptimizationFlag());
  }
  /// Onnx Runtime Execution Provider Flags
  if (aCommand == fDnnlFlagCmd)
  {
    cv = fDnnlFlagCmd->ConvertToString(fInference->GetDnnlFlag());
  } 
  else if (aCommand == fOpenVinoFlagCmd)
  {
    cv = fOpenVinoFlagCmd->ConvertToString(fInference->GetOpenVinoFlag());
  }
  else if (aCommand == fCudaFlagCmd)
  {
    cv = fCudaFlagCmd->ConvertToString(fInference->GetCudaFlag());
  }
  else if (aCommand == fTensorrtFlagCmd)
  {
    cv = fTensorrtFlagCmd->ConvertToString(fInference->GetTensorrtFlag());
  } 

  ///// Execution Provider Options
  /// oneDNN
  if (aCommand == fDnnlEnableCpuMemArenaCmd)
  {
    cv = fDnnlEnableCpuMemArenaCmd->ConvertToString(fInference->GetDnnlEnableCpuMemArena());
  }

  /// Cuda
  if (aCommand == fCudaDeviceIdCmd)
  {
    cv = fCudaDeviceIdCmd->ConvertToString(fInference->GetCudaDeviceId());
  }
  else if (aCommand == fCudaGpuMemLimitCmd)
  {
    cv = fCudaGpuMemLimitCmd->ConvertToString(fInference->GetCudaGpuMemLimit());
  }
  else if (aCommand == fCudaArenaExtendedStrategyCmd)
  {
    cv = fCudaArenaExtendedStrategyCmd->ConvertToString(fInference->GetCudaArenaExtendedStrategy());
  }
  else if (aCommand == fCudaCudnnConvAlgoSearchCmd)
  {
    cv = fCudaCudnnConvAlgoSearchCmd->ConvertToString(fInference->GetCudaCudnnConvAlgoSearch());
  }
  else if (aCommand == fCudaDoCopyInDefaultStreamCmd)
  {
    cv = fCudaDoCopyInDefaultStreamCmd->ConvertToString(fInference->GetCudaDoCopyInDefaultStream());
  }
  else if (aCommand == fCudaCudnnConvUseMaxWorkspaceCmd)
  {
    cv = fCudaCudnnConvUseMaxWorkspaceCmd->ConvertToString(fInference->GetCudaCudnnConvUseMaxWorkspace());
  }

  /// TensorRT
  if (aCommand == fTrtDeviceIdCmd)
  {
    cv = fTrtDeviceIdCmd->ConvertToString(fInference->GetTrtDeviceId());
  }
  else if (aCommand == fTrtMaxWorkspaceSizeCmd)
  {
    cv = fTrtMaxWorkspaceSizeCmd->ConvertToString(fInference->GetTrtMaxWorkspaceSize());
  }
  else if (aCommand == fTrtMaxPartitionIterationsCmd)
  {
    cv = fTrtMaxPartitionIterationsCmd->ConvertToString(fInference->GetTrtMaxPartitionIterations());
  }
  else if (aCommand == fTrtMinSubgraphSizeCmd)
  {
    cv = fTrtMinSubgraphSizeCmd->ConvertToString(fInference->GetTrtMinSubgraphSize());
  }
  else if (aCommand == fTrtFp16EnableCmd)
  {
    cv = fTrtFp16EnableCmd->ConvertToString(fInference->GetTrtFp16Enable());
  }
  else if (aCommand == fTrtInt8EnableCmd)
  {
    cv = fTrtInt8EnableCmd->ConvertToString(fInference->GetTrtInt8Enable());
  }
  else if (aCommand == fTrtInt8UseNativeCalibrationTableCmd)
  {
    cv = fTrtInt8UseNativeCalibrationTableCmd->ConvertToString(fInference->GetTrtInt8UseNativeCalibrationTable());
  }
  else if (aCommand == fTrtEngineCacheEnableCmd)
  {
    cv = fTrtEngineCacheEnableCmd->ConvertToString(fInference->GetTrtEngineCacheEnable());
  }
  else if (aCommand == fTrtEngineCachePathCmd)
  {
    cv = fTrtEngineCachePathCmd->ConvertToString(fInference->GetTrtEngineCachePath());
  }
  else if (aCommand == fTrtDumpSubgraphsCmd)
  {
    cv = fTrtDumpSubgraphsCmd->ConvertToString(fInference->GetTrtDumpSubgraphs());
  }

  /// OpenVINO

  if (aCommand == fOVDeviceTypeCmd)
  {
    cv = fOVDeviceTypeCmd->ConvertToString(fInference->GetOVDeviceType());
  }
  else if (aCommand == fOVEnableVpuFastCompileCmd)
  {
    cv = fOVEnableVpuFastCompileCmd->ConvertToString(fInference->GetOVEnableVpuFastCompile());
  }
  else if (aCommand == fOVDeviceIdCmd)
  {
    cv = fOVDeviceIdCmd->ConvertToString(fInference->GetOVDeviceId());
  }
  else if (aCommand == fOVNumOfThreadsCmd)
  {
    cv = fOVNumOfThreadsCmd->ConvertToString(fInference->GetOVNumOfThreads());
  }
  else if (aCommand == fOVUseCompiledNetworkCmd)
  {
    cv = fOVUseCompiledNetworkCmd->ConvertToString(fInference->GetOVUseCompiledNetwork());
  }
  else if (aCommand == fOVBlobDumpPathCmd)
  {
    cv = fOVBlobDumpPathCmd->ConvertToString(fInference->GetOVBlobDumpPath());
  }
  else if (aCommand == fOVEnableOpenCLThrottlingCmd)
  {
    cv = fOVEnableOpenCLThrottlingCmd->ConvertToString(fInference->GetOVEnableOpenCLThrottling());
  }

  /// ORT EP END
  if(aCommand == fMeshNbRhoCellsCmd)
  {
    cv = fMeshNbRhoCellsCmd->ConvertToString(fInference->GetMeshNbOfCells()[0]);
  }
  else if(aCommand == fMeshNbPhiCellsCmd)
  {
    cv = fMeshNbPhiCellsCmd->ConvertToString(fInference->GetMeshNbOfCells()[1]);
  }
  else if(aCommand == fMeshNbZCellsCmd)
  {
    cv = fMeshNbZCellsCmd->ConvertToString(fInference->GetMeshNbOfCells()[2]);
  }
  else if(aCommand == fMeshSizeRhoCellsCmd)
  {
    cv = fMeshSizeRhoCellsCmd->ConvertToString(fInference->GetMeshSizeOfCells()[0]);
  }
  else if(aCommand == fMeshSizeZCellsCmd)
  {
    cv = fMeshSizeZCellsCmd->ConvertToString(fInference->GetMeshSizeOfCells()[2]);
  }

  return cv;
}

#endif