// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4ParticleChangeForTransport.cc,v 2.8 1998/11/10 19:05:18 japost Exp $
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
//   Implemented for the new scheme                 10 May. 1998  H.Kurahige
//   Correct tratment of fpNextTouchable            12 May. 1998  H.Kurashige
// --------------------------------------------------------------

#include "G4ParticleChangeForTransport.hh"
#include "G4VTouchable.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4TrackFastVector.hh"
#include "G4DynamicParticle.hh"

G4ParticleChangeForTransport::G4ParticleChangeForTransport():G4ParticleChange()
{
  if (verboseLevel>2) {
    G4cerr << "G4ParticleChangeForTransport::G4ParticleChangeForTransport() " << endl;
  }
}

G4ParticleChangeForTransport::~G4ParticleChangeForTransport() 
{
  if (verboseLevel>2) {
    G4cerr << "G4ParticleChangeForTransport::~G4ParticleChangeForTransport() " << endl;
  }
}


G4ParticleChangeForTransport::G4ParticleChangeForTransport(const G4ParticleChangeForTransport &right):G4ParticleChange(right)
{
  if (verboseLevel>0) {
    G4cerr << "G4ParticleChangeForTransport::  copy constructor is called " << endl;
  }
  theTouchableChange = right.theTouchableChange;
}

// assignemnt operator
G4ParticleChangeForTransport & G4ParticleChangeForTransport::operator=(const G4ParticleChangeForTransport &right)
{
   if (verboseLevel>1) {
    G4cerr << "G4ParticleChangeForTransport:: assignment operator is called " << endl;
   }
   if (this != &right)
   {
      theListOfSecondaries = right.theListOfSecondaries;
      theSizeOftheListOfSecondaries = right.theSizeOftheListOfSecondaries;
      theNumberOfSecondaries = right.theNumberOfSecondaries;
      theStatusChange = right.theStatusChange;
      theTouchableChange = right.theTouchableChange;
      theMomentumChange = right.theMomentumChange;
      thePolarizationChange = right.thePolarizationChange;
      thePositionChange = right.thePositionChange;
      theTimeChange = right.theTimeChange;
      theEnergyChange = right.theEnergyChange;
      theTrueStepLength = right.theTrueStepLength;
      theLocalEnergyDeposit = right.theLocalEnergyDeposit;
      theSteppingControlFlag = right.theSteppingControlFlag;
   }
   return *this;
}

//----------------------------------------------------------------
// methods for updating G4Step 
//

G4Step* G4ParticleChangeForTransport::UpdateStepForAtRest(G4Step* pStep)
{ 
  // Nothing happens for AtRestDoIt
  if (verboseLevel>0) {
    G4cerr << "G4ParticleChangeForTransport::UpdateStepForAtRest() is called" << endl; 
    G4cerr << " Nothing happens for this method " << endl; 
  }
  //  Update the G4Step specific attributes 
  return UpdateStepInfo(pStep);
}


G4Step* G4ParticleChangeForTransport::UpdateStepForAlongStep(G4Step* pStep)
{
  // copy of G4ParticleChange::UpdateStepForAlongStep
  //  i.e. no effect for touchable

  // A physics process always calculates the final state of the
  // particle relative to the initial state at the beginning
  // of the Step, i.e., based on information of G4Track (or
  // equivalently the PreStepPoint). 
  // So, the differences (delta) between these two states have to be
  // calculated and be accumulated in PostStepPoint. 
  
  // Take note that the return type of GetMomentumChange is a
  // pointer to G4ParticleMomentum. Also it is a normalized 
  // momentum vector.

  G4StepPoint* pPreStepPoint  = pStep->GetPreStepPoint(); 
  G4StepPoint* pPostStepPoint = pStep->GetPostStepPoint(); 
  G4Track*     aTrack  = pStep->GetTrack();
  G4double     mass = mass = aTrack->GetDynamicParticle()->GetMass();
 
  // uodate kinetic energy
  //  now assume that no energy change in transportation
  //  However it is not true in electric fields
  //  Case for changing energy will be implemented in future

  
  // update momentum direction and energy
  if (isMomentumChanged) {
    G4double energy;
    energy= pPostStepPoint->GetKineticEnergy() 
                 + (theEnergyChange - pPreStepPoint->GetKineticEnergy()); 

    // calculate new momentum
    G4ThreeVector pMomentum =  pPostStepPoint->GetMomentum() 
                     + ( CalcMomentum(theEnergyChange, theMomentumChange, mass)
	                  - pPreStepPoint->GetMomentum());
    G4double      tMomentum_inv = 1.0 / pMomentum.mag();
    pPostStepPoint->SetMomentumDirection(pMomentum*tMomentum_inv);
    pPostStepPoint->SetKineticEnergy( energy );
  }

  // stop case should not occur
  //pPostStepPoint->SetMomentumDirection(G4ThreeVector(1., 0., 0.));


  // update polarization
  //pPostStepPoint->AddPolarization( thePolarizationChange
  //				   - pPreStepPoint->GetPolarization());
      
  // update position and time
  pPostStepPoint->AddPosition( thePositionChange 
			       - pPreStepPoint->GetPosition() );
  pPostStepPoint->AddGlobalTime( theTimeChange
				 - pPreStepPoint->GetGlobalTime());
  pPostStepPoint->AddLocalTime( theTimeChange 
				 - pPreStepPoint->GetGlobalTime()); 
  pPostStepPoint->AddProperTime( theProperTimeChange 
				 - pPreStepPoint->GetProperTime());

#ifdef G4VERBOSE
  if (debugFlag) CheckIt(*aTrack);
#endif

  //  Update the G4Step specific attributes 
  //pStep->SetStepLength( theTrueStepLength );
  //  pStep->AddTotalEnergyDeposit( theLocalEnergyDeposit );
  pStep->SetControlFlag( theSteppingControlFlag );
  return pStep;
  //  return UpdateStepInfo(pStep);
}

G4Step* G4ParticleChangeForTransport::UpdateStepForPostStep(G4Step* pStep)
{ 
  // A physics process always calculates the final state of the particle

  G4StepPoint* pPreStepPoint  = pStep->GetPreStepPoint(); 
  G4StepPoint* pPostStepPoint = pStep->GetPostStepPoint(); 
  G4Track*     aTrack  = pStep->GetTrack();

  // update next touchable 
  // (touchable can be changed only at PostStepDoIt) 
  pPostStepPoint->SetTouchable( theTouchableChange );

  // It used to call base class's method 
  //   - but this would copy uninitialised data members
  // return G4ParticleChange::UpdateStepForPostStep(pStep);

  // Copying what the base class does would instead 
  //   - also not useful
  // return G4VParticleChange::UpdateStepInfo(pStep);

  return pStep; 
}

//----------------------------------------------------------------
// methods for printing messages  
//

void G4ParticleChangeForTransport::DumpInfo() const
{
// use base-class DumpInfo
  G4ParticleChange::DumpInfo();

  G4cout.precision(3);
  G4cout << "        Touchable (pointer) : " 
       << setw(20) << theTouchableChange
       << endl; 
}





