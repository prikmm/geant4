// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: ExN05EventActionMessenger.hh,v 2.1 1998/07/12 02:42:14 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//

#ifndef ExN05EventActionMessenger_h
#define ExN05EventActionMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
class G4UIdirectory;
class G4UIcmdWithABool;

class ExN05EventAction;

class ExN05EventActionMessenger: public G4UImessenger
{
  public:
    ExN05EventActionMessenger(ExN05EventAction* EA);
    void SetNewValue(G4UIcommand* command, G4String newValues);
  private:
    ExN05EventAction* EventAction;
    G4UIdirectory*    eventDirectory;
    G4UIcmdWithABool* drawEventCmd;
};

#endif

