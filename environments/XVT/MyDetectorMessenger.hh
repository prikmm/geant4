// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: MyDetectorMessenger.hh,v 2.2 1998/07/13 17:29:41 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//

#ifndef MyDetectorMessenger_h
#define MyDetectorMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

#include "G4ios.hh"
#ifdef WIN32
#  include <Strstrea.h>
#else
#  include <strstream.h>
#endif

class G4UIcommand;
class MyDetectorConstruction;

class MyDetectorMessenger: public G4UImessenger
{
  public:
    MyDetectorMessenger(MyDetectorConstruction * myDC);
    void SetNewValue(G4UIcommand * command,G4String newValues);
  private:
    MyDetectorConstruction * myDetector;
};

#endif

