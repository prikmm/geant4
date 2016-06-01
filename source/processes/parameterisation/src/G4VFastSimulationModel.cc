// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VFastSimulationModel.cc,v 2.1 1998/10/13 09:54:45 mora Exp $
// GEANT4 tag $Name: geant4-00 $
//
// $Id:
//---------------------------------------------------------------
//
//  G4VFastSimulationModel.cc
//
//  Description:
//    Base class for fast simulation models.
//
//  History:
//    Oct 97: Verderi && MoraDeFreitas - First Implementation.
//
//---------------------------------------------------------------


#include "G4VFastSimulationModel.hh"
#include "G4FastSimulationManager.hh"

//
// Simple constructor.
//
G4VFastSimulationModel::
G4VFastSimulationModel(const G4String& aName)  : theModelName(aName) {}

//
// Constructor for beginners. We do all the job, no matters...
//
G4VFastSimulationModel::
G4VFastSimulationModel(const G4String& aName,
		       G4Envelope* anEnvelope,
		       G4bool IsUnique) : theModelName(aName)
{
  // Retrieves the Fast Simulation Manager ou creates one 
  // if needed.
  G4FastSimulationManager* theFastSimulationManager;
  if ((theFastSimulationManager=anEnvelope->GetFastSimulationManager())
      ==NULL) theFastSimulationManager= 
		new G4FastSimulationManager(anEnvelope,IsUnique);
  // adds this model to the Fast Simulation Manager.
  theFastSimulationManager->AddFastSimulationModel(this);
}
