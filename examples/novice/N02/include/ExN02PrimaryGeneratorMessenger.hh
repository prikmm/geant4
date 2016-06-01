// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: ExN02PrimaryGeneratorMessenger.hh,v 1.3 1998/10/09 14:31:05 japost Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 

#ifndef ExN02PrimaryGeneratorMessenger_h
#define ExN02PrimaryGeneratorMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class ExN02PrimaryGeneratorAction;
class G4UIcmdWithAString;

class ExN02PrimaryGeneratorMessenger: public G4UImessenger
{
  public:
    ExN02PrimaryGeneratorMessenger(ExN02PrimaryGeneratorAction* myGun);
    ~ExN02PrimaryGeneratorMessenger();
    
    void SetNewValue(G4UIcommand * command,G4String newValues);
    
  private:
    ExN02PrimaryGeneratorAction* myAction;
 
    G4UIcmdWithAString*       RndmCmd;
};

#endif

