// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4StopDeexcitation.cc,v 2.2 1998/07/14 12:35:32 jwellisc Exp $
// GEANT4 tag $Name: geant4-00 $
//
// -------------------------------------------------------------------
//      GEANT 4 class file --- Copyright CERN 1998
//      CERN Geneva Switzerland
//
//      For information related to this code contact:
//      CERN, IT Division, ASD group
//
//      File name:     G4StopDeexcitation
//
//      Author:        Maria Grazia Pia (pia@genova.infn.it)
// 
//      Creation date: 8 May 1998
//
//      Modifications: 
// -------------------------------------------------------------------


#include "G4StopDeexcitation.hh"
#include <rw/tpordvec.h>
#include <rw/tvordvec.h>
#include <rw/cstring.h>

#include "globals.hh"
#include "Randomize.hh"
#include "G4ParticleTypes.hh"
#include "G4DynamicParticleVector.hh"
#include "G4NucleiPropertiesTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ThreeVector.hh"


// Constructor

G4StopDeexcitation::G4StopDeexcitation(G4StopDeexcitationAlgorithm* algorithm)
  
{
  _algorithm = algorithm;
}


// Destructor

G4StopDeexcitation::~G4StopDeexcitation()
{
  delete _algorithm;
}

G4DynamicParticleVector* G4StopDeexcitation::DoBreakUp(G4double A, G4double Z, 
						       G4double excitation, 
						       const G4ThreeVector& p) const
{
  if (_algorithm != 0) 
    {
      return _algorithm->BreakUp(A,Z,excitation,p);
    }
  else 
    return 0;
}
