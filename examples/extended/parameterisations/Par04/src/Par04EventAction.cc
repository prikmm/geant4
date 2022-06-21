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

#include "Par04EventAction.hh"
#include <CLHEP/Units/SystemOfUnits.h>   // for GeV
#include <CLHEP/Vector/ThreeVector.h>    // for Hep3Vector
#include <stddef.h>                      // for size_t
#include <G4Exception.hh>                // for G4Exception, G4ExceptionDesc...
#include <G4ExceptionSeverity.hh>        // for FatalException
#include <G4GenericAnalysisManager.hh>   // for G4GenericAnalysisManager
#include <G4PrimaryParticle.hh>          // for G4PrimaryParticle
#include <G4PrimaryVertex.hh>            // for G4PrimaryVertex
#include <G4SystemOfUnits.hh>            // for GeV
#include <G4THitsCollection.hh>          // for G4THitsCollection
#include <G4ThreeVector.hh>              // for G4ThreeVector
#include <G4Timer.hh>                    // for G4Timer
#include <G4UserEventAction.hh>          // for G4UserEventAction
#include <algorithm>                     // for max
#include <ostream>                       // for basic_ostream::operator<<
#include "G4AnalysisManager.hh"          // for G4AnalysisManager
#include "G4Event.hh"                    // for G4Event
#include "G4EventManager.hh"             // for G4EventManager
#include "G4HCofThisEvent.hh"            // for G4HCofThisEvent
#include "G4SDManager.hh"                // for G4SDManager
#include "Par04DetectorConstruction.hh"  // for Par04DetectorConstruction
#include "Par04Hit.hh"                   // for Par04Hit, Par04HitsCollection
#ifdef USE_ROOT
#include "TSystem.h"
#endif
#ifdef USE_CUDA
//#include "cuda.h"
#include "cuda_runtime_api.h"
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04EventAction::Par04EventAction(Par04DetectorConstruction* aDetector)
  : G4UserEventAction()
  , fHitCollectionID(-1)
  , fTimer()
  , fDetector(aDetector)
{
  fCellNbRho = aDetector->GetMeshNbOfCells().x();
  fCellNbPhi = aDetector->GetMeshNbOfCells().y();
  fCellNbZ   = aDetector->GetMeshNbOfCells().z();
  fCalEdep.reserve(fCellNbRho * fCellNbPhi * fCellNbZ);
  fCalRho.reserve(fCellNbRho * fCellNbPhi * fCellNbZ);
  fCalPhi.reserve(fCellNbRho * fCellNbPhi * fCellNbZ);
  fCalZ.reserve(fCellNbRho * fCellNbPhi * fCellNbZ);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04EventAction::~Par04EventAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04EventAction::BeginOfEventAction(const G4Event*) 
{ 
  fTimer.Start();

  #ifdef USE_CUDA
  const G4float GPUtoMB = 1.f / (1024.f * 1024.f);
  size_t free, total;
  G4int num_gpus;
  cudaGetDeviceCount( &num_gpus );
  for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        //int id;
        //cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        usage_mem_gpu += total - free;
        //std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
  }
  //G4cout << "GPU mem before: " << usage_mem_gpu * GPUtoMB << G4endl;
  #endif
  #ifdef USE_ROOT
  const G4float CPUtoMB = 1.f / 1024.f;
  static ProcInfo_t info;
  gSystem->GetProcInfo(&info);
  tot_res_mem += info.fMemResident * CPUtoMB;
	tot_virt_mem += info.fMemVirtual * CPUtoMB;
  //G4cout << "CPU resident mem before: " << tot_res_mem << "\n"
  //       << "CPU virtual mem before: " << tot_virt_mem << G4endl;
  #endif
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04EventAction::EndOfEventAction(const G4Event* aEvent)
{
  fTimer.Stop();

  // Get hits collection ID (only once)
  if(fHitCollectionID == -1)
  {
    fHitCollectionID = G4SDManager::GetSDMpointer()->GetCollectionID("hits");
  }
  // Get hits collection
  auto hitsCollection =
    static_cast<Par04HitsCollection*>(aEvent->GetHCofThisEvent()->GetHC(fHitCollectionID));

  if(hitsCollection == nullptr)
  {
    G4ExceptionDescription msg;
    msg << "Cannot access hitsCollection ID " << fHitCollectionID;
    G4Exception("Par04EventAction::GetHitsCollection()", "MyCode0001", FatalException, msg);
  }

  // Get analysis manager
  auto analysisManager = G4AnalysisManager::Instance();
  // Retrieve only once detector dimensions
  if(fCellSizeZ == 0)
  {
    fCellSizeZ   = fDetector->GetMeshSizeOfCells().z();
    fCellSizeRho = fDetector->GetMeshSizeOfCells().x();
    fCellNbRho   = fDetector->GetMeshNbOfCells().x();
    fCellNbPhi   = fDetector->GetMeshNbOfCells().y();
    fCellNbZ     = fDetector->GetMeshNbOfCells().z();
  }

  // Retrieve information from primary vertex and primary particle
  // To calculate shower axis and entry point to the detector
  auto primaryVertex =
    G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetPrimaryVertex();
  auto primaryParticle   = primaryVertex->GetPrimary(0);
  G4double primaryEnergy = primaryParticle->GetTotalEnergy();
  // Estimate from vertex and particle direction the entry point to the detector
  // Calculate entrance point to the detector located at z = 0
  auto primaryDirection = primaryParticle->GetMomentumDirection();
  auto primaryEntrance =
    primaryVertex->GetPosition() - primaryVertex->GetPosition().z() * primaryDirection;

  // Resize back to initial mesh size
  fCalEdep.resize(fCellNbRho * fCellNbPhi * fCellNbZ);
  fCalRho.resize(fCellNbRho * fCellNbPhi * fCellNbZ);
  fCalPhi.resize(fCellNbRho * fCellNbPhi * fCellNbZ);
  fCalZ.resize(fCellNbRho * fCellNbPhi * fCellNbZ);

  // Fill histograms
  Par04Hit* hit                  = nullptr;
  G4double hitEn                 = 0;
  G4double totalEnergy           = 0;
  G4int hitZ                     = -1;
  G4int hitRho                   = -1;
  G4int hitPhi                   = -1;
  G4int hitType                  = -1;
  G4int numNonZeroThresholdCells = 0;
  G4double tDistance = 0., rDistance = 0.;
  G4double tFirstMoment = 0., tSecondMoment = 0.;
  G4double rFirstMoment = 0., rSecondMoment = 0.;
  for(size_t iHit = 0; iHit < hitsCollection->entries(); iHit++)
  {
    hit     = static_cast<Par04Hit*>(hitsCollection->GetHit(iHit));
    hitZ    = hit->GetZid();
    hitRho  = hit->GetRhoId();
    hitPhi  = hit->GetPhiId();
    hitEn   = hit->GetEdep();
    hitType = hit->GetType();
    if(hitEn > 0)
    {
      totalEnergy += hitEn;
      tDistance = hitZ * fCellSizeZ;
      rDistance = hitRho * fCellSizeRho;
      tFirstMoment += hitEn * tDistance;
      rFirstMoment += hitEn * rDistance;
      analysisManager->FillH1(4, tDistance, hitEn);
      analysisManager->FillH1(5, rDistance, hitEn);
      analysisManager->FillH1(10, hitType);
      if(hitEn > 0.0005)
      {  // e > 0.5 keV
        fCalEdep[numNonZeroThresholdCells] = hitEn;
        fCalRho[numNonZeroThresholdCells]  = hitRho;
        fCalPhi[numNonZeroThresholdCells]  = hitPhi;
        fCalZ[numNonZeroThresholdCells]    = hitZ;
        numNonZeroThresholdCells++;
      }
    }
  }
  tFirstMoment /= totalEnergy;
  rFirstMoment /= totalEnergy;
  analysisManager->FillH1(0, primaryEnergy / GeV);
  analysisManager->FillH1(1, totalEnergy / GeV);
  analysisManager->FillH1(2, totalEnergy / primaryEnergy);
  analysisManager->FillH1(3, fTimer.GetRealElapsed());
  analysisManager->FillH1(6, tFirstMoment);
  analysisManager->FillH1(7, rFirstMoment);
  // Resize to store only energy hits above threshold
  fCalEdep.resize(numNonZeroThresholdCells);
  fCalRho.resize(numNonZeroThresholdCells);
  fCalPhi.resize(numNonZeroThresholdCells);
  fCalZ.resize(numNonZeroThresholdCells);
  analysisManager->FillNtupleDColumn(0, primaryEnergy);
  analysisManager->FillNtupleDColumn(5, fTimer.GetRealElapsed());

  // Second loop over hits to calculate second moments
  for(size_t iHit = 0; iHit < hitsCollection->entries(); iHit++)
  {
    hit    = static_cast<Par04Hit*>(hitsCollection->GetHit(iHit));
    hitEn  = hit->GetEdep();
    hitZ   = hit->GetZid();
    hitRho = hit->GetRhoId();
    if(hitEn > 0)
    {
      tDistance = hitZ * fCellSizeZ;
      rDistance = hitRho * fCellSizeRho;
      tSecondMoment += hitEn * std::pow(tDistance - tFirstMoment, 2);
      rSecondMoment += hitEn * std::pow(rDistance - rFirstMoment, 2);
    }
  }
  tSecondMoment /= totalEnergy;
  rSecondMoment /= totalEnergy;
  analysisManager->FillH1(8, tSecondMoment);
  analysisManager->FillH1(9, rSecondMoment);


  // Memory Usage extraction
  #ifdef USE_ROOT
  const G4float CPUtoMB = 1.f / 1024.f;
  static ProcInfo_t info;
  gSystem->GetProcInfo(&info);
  G4double tot_res_mem_after = info.fMemResident * CPUtoMB;
	G4double tot_virt_mem_after = info.fMemVirtual * CPUtoMB;
  tot_res_mem += tot_res_mem_after - tot_res_mem;
	tot_virt_mem += tot_virt_mem_after - tot_virt_mem;
  analysisManager->FillNtupleDColumn(6, tot_res_mem);
  analysisManager->FillNtupleDColumn(7, tot_virt_mem);
  //G4cout << "CPU resident mem usage: " << tot_res_mem << "\n"
  //       << "CPU virtual mem usage: " << tot_virt_mem << G4endl;
  #endif
  #ifdef USE_CUDA
  const G4float GPUtoMB = 1.f / (1024.f * 1024.f);
  size_t free, total;
  size_t usage_mem_gpu_after = 0;
  G4int num_gpus;
  cudaGetDeviceCount( &num_gpus );
  for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        usage_mem_gpu_after += total - free;
        //std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
  }
  usage_mem_gpu += usage_mem_gpu_after - usage_mem_gpu;
  //G4cout << "GPU mem usage: " << usage_mem_gpu * GPUtoMB << G4endl;
  analysisManager->FillNtupleDColumn(8, usage_mem_gpu * GPUtoMB);
  #endif
  //
  analysisManager->AddNtupleRow();
}
