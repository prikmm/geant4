// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4PiMinusStopC.hh,v 2.2 1998/10/01 19:30:12 pia Exp $
// GEANT4 tag $Name: geant4-00 $
//
// -------------------------------------------------------------------
//      GEANT 4 class file --- Copyright CERN 1998
//      CERN Geneva Switzerland
//
//      For information related to this code contact:
//      CERN, IT Division, ASD group
//
//      File name:     G4PiMinusStopC.hh 
//
//      Author:        Maria Grazia Pia (pia@genova.infn.it)
// 
//      Creation date: 18 May 1998
//
//      Modifications: 
// -------------------------------------------------------------------

#ifndef G4PIMINUSSTOPC_HH
#define G4PIMINUSSTOPC_HH 

#include <rw/tpordvec.h>
#include <rw/tvordvec.h>
#include <rw/cstring.h>

#include "G4PiMinusStopMaterial.hh"
#include "globals.hh"
#include "G4LorentzVector.hh"
//#include "G4String.hh"
//#include "G4NucleiPropertiesTable.hh"

class G4PiMinusStopC : public G4PiMinusStopMaterial
{  

private:

  // Hide assignment operator as private 
  G4PiMinusStopC& operator=(const G4PiMinusStopC &right);

  // Copy constructor
  G4PiMinusStopC(const G4PiMinusStopC& );

public:

  // Constructor
  G4PiMinusStopC();

  // Destructor
  virtual ~G4PiMinusStopC();

  // Number of final nucleons, out of generated absorption products
  virtual G4double FinalNucleons();


private:

  static G4int eKinEntries;
  static G4int angleEntries;

  static G4double npRatio;
 
  static G4double nFinalNucleons;

  static G4double eKinData[21];
  static G4double eKin[22];

  static G4double angleData[7];
  static G4double angle[8];

  G4double _clusterSize;

};

#endif


