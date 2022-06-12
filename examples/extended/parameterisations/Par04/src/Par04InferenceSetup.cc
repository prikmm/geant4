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
#include "Par04InferenceSetup.hh"

#include "Par04InferenceInterface.hh"
#ifdef USE_INFERENCE_ONNX
#include "Par04OnnxInference.hh"
#include <variant>
#endif
#ifdef USE_INFERENCE_LWTNN
#include "Par04LwtnnInference.hh"
#endif
#include "G4RotationMatrix.hh"
#include "CLHEP/Random/RandGauss.h"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04InferenceSetup::Par04InferenceSetup()
  : fInferenceMessenger(new Par04InferenceMessenger(this))
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04InferenceSetup::~Par04InferenceSetup() {
  delete fInferenceMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool Par04InferenceSetup::IfTrigger(G4double aEnergy)
{
  /// Energy of electrons used in training dataset
  if(aEnergy > 1 * CLHEP::GeV || aEnergy < 1024 * CLHEP::GeV)
    return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04InferenceSetup::SetInferenceLibrary(G4String aName)
{
  fInferenceLibrary = aName;

#ifdef USE_INFERENCE_ONNX
  if(fInferenceLibrary == "ONNX"){
  
    //OpenVINO 
    std::vector<std::variant<const char *, int>> openvino_options {
        fOVDeviceType.c_str(),
        fOVEnableVpuFastCompile,
        fOVDeviceId.c_str(),
        fOVNumOfThreads,
        fOVUseCompiledNetwork,
        fOVBlobDumpPath.c_str(),
    };
    // TensorRT
    std::vector<const char *> trt_keys{
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
        fTrtDeviceId.c_str(),                     
        fTrtMaxWorkspaceSize.c_str(),            
        fTrtMaxPartitionIterations.c_str(),                    
        fTrtMinSubgraphSize.c_str(),                     
        fTrtFp16Enable.c_str(),                     
        fTrtInt8Enable.c_str(),                     
        fTrtInt8UseNativeCalibrationTable.c_str(),                     
        fTrtEngineCacheEnable.c_str(),                     
        fTrtEngineCachePath.c_str(), 
        fTrtDumpSubgraphs.c_str(),                
    };
    // CUDA
    std::vector<const char *> cuda_keys{
        "device_id",
        "gpu_mem_limit",
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
    };
    std::vector<const char *> cuda_values{
        fCudaDeviceId.c_str(),
        fCudaGpuMemLimit.c_str(),
        fCudaArenaExtendedStrategy.c_str(),
        fCudaCudnnConvAlgoSearch.c_str(),
        fCudaDoCopyInDefaultStream.c_str(),
        fCudaCudnnConvUseMaxWorkspace.c_str(),
    };
    
    fInferenceInterface = std::unique_ptr<Par04InferenceInterface>(
      new Par04OnnxInference(fModelPathName, fProfileFlag, fOptimizationFlag, fIntraOpNumThreads,
                             fDnnlFlag, fOpenVinoFlag, fCudaFlag, fTensorrtFlag,
                             fDnnlEnableCpuMemArena,
                             openvino_options,
                             cuda_keys, cuda_values,
                             trt_keys, trt_values));
  
  }
#endif
#ifdef USE_INFERENCE_LWTNN
  if(fInferenceLibrary == "LWTNN")
    fInferenceInterface =
      std::unique_ptr<Par04InferenceInterface>(new Par04LwtnnInference(fModelPathName));
#endif
  CheckInferenceLibrary();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04InferenceSetup::CheckInferenceLibrary()
{
  G4String msg = "Please choose inference library from available libraries (";
#ifdef USE_INFERENCE_ONNX
  msg += "ONNX,";
#endif
#ifdef USE_INFERENCE_LWTNN
  msg += "LWTNN";
#endif
  if(fInferenceInterface == nullptr)
    G4Exception("Par04InferenceSetup::CheckInferenceLibrary()", "InvalidSetup", FatalException,
                (msg + "). Current name: " + fInferenceLibrary).c_str());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04InferenceSetup::GetEnergies(std::vector<G4double>& aEnergies, G4double aInitialEnergy,
                                      G4float aInitialAngle)
{
  // First check if inference library was set correctly
  CheckInferenceLibrary();
  // size represents the size of the output vector
  int size = fMeshNumber.x() * fMeshNumber.y() * fMeshNumber.z();

  // randomly sample from a gaussian distribution in the latent space
  std::vector<G4float> genVector(fSizeLatentVector + fSizeConditionVector, 0);
  for(int i = 0; i < fSizeLatentVector; ++i)
  {
    genVector[i] = CLHEP::RandGauss::shoot(0., 1.);
  }

  // Vector of condition
  // this is application specific it depdens on what the model was condition on
  // and it depends on how the condition values were encoded at the training time
  // in this example the energy of each particle is normlaized to the highest
  // energy in the considered range (1GeV-500GeV)
  // the angle is also is normlaized to the highest angle in the considered range
  // (0-90 in dergrees)
  // the model in this example was trained on two detector geometries PBW04
  // and SiW  a one hot encoding vector is used to represent the geometry with
  // [0,1] for PBW04 and [1,0] for SiW
  // 1.energy
  genVector[fSizeLatentVector] = aInitialEnergy / fMaxEnergy;
  // 2. angle
  genVector[fSizeLatentVector + 1] = (aInitialAngle / (CLHEP::deg)) / fMaxAngle;
  // 3.geometry
  genVector[fSizeLatentVector + 2] = 0;
  genVector[fSizeLatentVector + 3] = 1;

  // Run the inference
  fInferenceInterface->RunInference(genVector, aEnergies, size);

  // After the inference rescale back to the initial energy (in this example the
  // energies of cells were normalized to the energy of the particle)
  for(int i = 0; i < size; ++i)
  {
    aEnergies[i] = aEnergies[i] * aInitialEnergy;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04InferenceSetup::GetPositions(std::vector<G4ThreeVector>& aPositions, G4ThreeVector pos0,
                                       G4ThreeVector direction)
{
  aPositions.resize(fMeshNumber.x() * fMeshNumber.y() * fMeshNumber.z());

  // Calculate rotation matrix along the particle momentum direction
  // It will rotate the shower axes to match the incoming particle direction
  G4RotationMatrix rotMatrix = G4RotationMatrix();
  double particleTheta       = direction.theta();
  double particlePhi         = direction.phi();
  rotMatrix.rotateZ(-particlePhi);
  rotMatrix.rotateY(-particleTheta);
  G4RotationMatrix rotMatrixInv = CLHEP::inverseOf(rotMatrix);

  int cpt = 0;
  for(G4int iCellR = 0; iCellR < fMeshNumber.x(); iCellR++)
  {
    for(G4int iCellPhi = 0; iCellPhi < fMeshNumber.y(); iCellPhi++)
    {
      for(G4int iCellZ = 0; iCellZ < fMeshNumber.z(); iCellZ++)
      {
        aPositions[cpt] =
          pos0 +
          rotMatrixInv *
            G4ThreeVector((iCellR + 0.5) * fMeshSize.x() *
                            std::cos((iCellPhi + 0.5) * 2 * CLHEP::pi / fMeshNumber.y() - CLHEP::pi),
                          (iCellR + 0.5) * fMeshSize.x() *
                            std::sin((iCellPhi + 0.5) * 2 * CLHEP::pi / fMeshNumber.y() - CLHEP::pi),
                          (iCellZ + 0.5) * fMeshSize.z());
        cpt++;
      }
    }
  }
}

#endif
