// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4PiMinusStopN.hh,v 2.1 1998/07/12 03:07:15 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//
// -------------------------------------------------------------------
//      GEANT 4 class file --- Copyright CERN 1998
//      CERN Geneva Switzerland
//
//      For information related to this code contact:
//      CERN, IT Division, ASD group
//
//      File name:     G4PiMinusStopN.hh 
//
//      Author:        Maria Grazia Pia (pia@genova.infn.it)
// 
//      Creation date: 18 May 1998
//
//      Modifications: 
// -------------------------------------------------------------------

#ifndef G4PIMINUSSTOPN_HH
#define G4PIMINUSSTOPN_HH 

#include <rw/tpordvec.h>
#include <rw/tvordvec.h>
#include <rw/cstring.h>

#include "G4PiMinusStopMaterial.hh"
#include "globals.hh"
#include "G4LorentzVector.hh"

class G4PiMinusStopN : public G4PiMinusStopMaterial
{  

private:

  // Hide assignment operator as private 
  G4PiMinusStopN& operator=(const G4PiMinusStopN &right);

  // Copy constructor
  G4PiMinusStopN(const G4PiMinusStopN& );

public:

  // Constructor
  G4PiMinusStopN();

  // Destructor
  virtual ~G4PiMinusStopN();

  // Definitions of absorption products
  //  virtual RWTValOrderedVector<G4String>* DefinitionVector();

  // 4-vectors of absorption products
  //  virtual RWTPtrOrderedVector<G4LorentzVector>* P4Vector();

  // Number of final nucleons, out of generated absorption products
  virtual G4double FinalNucleons();


private:

  static G4int eKinEntries;
  static G4int angleEntries;

  static G4double npRatio;
 
  static G4double nFinalNucleons;

  static G4double eMaxTot;

  static G4double eKinData[10];
  static G4double eKin[11];

  static G4double angleData[7];
  static G4double angle[8];

  G4double _clusterSize;

};

#endif


