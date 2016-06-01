// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4ParticleChangeForDecay.cc,v 2.2 1998/07/20 06:11:57 kurasige Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// --------------------------------------------------------------
//	GEANT 4 class implementation file 
//
//	For information related to this code contact:
//	CERN, CN Division, ASD Group
//	
//	
// ------------------------------------------------------------
//   Implemented for the new scheme                 23 Mar. 1998  H.Kurahige
//   Remove modification of energy/momentum         20 Jul, 1998  H.Kurashige
// --------------------------------------------------------------

#include "G4ParticleChangeForDecay.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4TrackFastVector.hh"
#include "G4DynamicParticle.hh"

G4ParticleChangeForDecay::G4ParticleChangeForDecay():G4VParticleChange()
{
#ifdef G4VERBOSE
  if (verboseLevel>2) {
    G4cerr << "G4ParticleChangeForDecay::G4ParticleChangeForDecay() " << endl;
  }
#endif
}

G4ParticleChangeForDecay::~G4ParticleChangeForDecay() 
{
#ifdef G4VERBOSE
  if (verboseLevel>2) {
    G4cerr << "G4ParticleChangeForDecay::~G4ParticleChangeForDecay() " << endl;
  }
#endif
}

// copy and assignment operators are implemented as "shallow copy"
G4ParticleChangeForDecay::G4ParticleChangeForDecay(const G4ParticleChangeForDecay &right)
{
   *this = right;
}


G4ParticleChangeForDecay & G4ParticleChangeForDecay::operator=(const G4ParticleChangeForDecay &right)
{
   if (this != &right)
   {
      theListOfSecondaries = right.theListOfSecondaries;
      theSizeOftheListOfSecondaries = right.theSizeOftheListOfSecondaries;
      theNumberOfSecondaries = right.theNumberOfSecondaries;
      theStatusChange = right.theStatusChange;
      theTrueStepLength = right.theTrueStepLength;
      theLocalEnergyDeposit = right.theLocalEnergyDeposit;
      theSteppingControlFlag = right.theSteppingControlFlag;
   }
   return *this;
}

G4bool G4ParticleChangeForDecay::operator==(const G4ParticleChangeForDecay &right) const
{
   return ((G4VParticleChange *)this == (G4VParticleChange *) &right);
}

G4bool G4ParticleChangeForDecay::operator!=(const G4ParticleChangeForDecay &right) const
{
   return ((G4VParticleChange *)this != (G4VParticleChange *) &right);
}

//----------------------------------------------------------------
// methods for Initialization
//
void G4ParticleChangeForDecay::Initialize(const G4Track& track)
{
  // use base class's method at first
  G4VParticleChange::Initialize(track);

  // set Time e equal to those of the parent track
  theTimeChange          = track.GetGlobalTime();
}

//----------------------------------------------------------------
// methods for updating G4Step 
//

G4Step* G4ParticleChangeForDecay::UpdateStepForPostStep(G4Step* pStep)
{ 
  // A physics process always calculates the final state of the particle
  G4StepPoint* pPostStepPoint = pStep->GetPostStepPoint(); 

  //  Update the G4Step specific attributes 
  return UpdateStepInfo(pStep);
}


G4Step* G4ParticleChangeForDecay::UpdateStepForAtRest(G4Step* pStep)
{ 
  // A physics process always calculates the final state of the particle

  G4StepPoint* pPreStepPoint  = pStep->GetPreStepPoint(); 
  G4StepPoint* pPostStepPoint = pStep->GetPostStepPoint(); 
 
  // update time
  pPostStepPoint->SetGlobalTime( theTimeChange  );
  pPostStepPoint->AddLocalTime(  theTimeChange 
                                 - pPreStepPoint->GetGlobalTime());
  pPostStepPoint->AddProperTime( theTimeChange  
                                 - pPreStepPoint->GetGlobalTime());

  //  Update the G4Step specific attributes 
  return UpdateStepInfo(pStep);
}

void G4ParticleChangeForDecay::DumpInfo() const
{
// Show header
  G4VParticleChange::DumpInfo();

  G4cout.precision(3);
  G4cout << "        Time (ns)           : " 
       << setw(20) << theTimeChange/ns
       << endl;
}





