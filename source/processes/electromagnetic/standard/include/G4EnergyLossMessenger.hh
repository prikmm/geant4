// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4EnergyLossMessenger.hh,v 2.3 1998/10/20 12:23:47 maire Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#ifndef G4EnergyLossMessenger_h
#define G4EnergyLossMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class G4UIcommand;
class G4UIcmdWithABool;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class G4EnergyLossMessenger: public G4UImessenger
{
  public:
  
    G4EnergyLossMessenger();
   ~G4EnergyLossMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
  private:
      
    G4UIcmdWithABool*          RndmStepCmd;
    G4UIcmdWithABool*          EnlossFlucCmd;
    G4UIcommand*               StepFuncCmd;
};

#endif

