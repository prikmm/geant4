// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4ExcitedBaryons.hh,v 2.1 1998/07/13 17:19:02 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// ------------------------------------------------------------
//      GEANT 4 class header file
//
//      For information related to this code contact:
//      CERN, CN Division, ASD group
//      History: first implementation, based on object model of
//      Hisaya Kurashige, 27 June 1998
// ----------------------------------------------------------------


#ifndef G4ExcitedBaryons_h
#define G4ExcitedBaryons_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "G4VShortLivedParticle.hh"

// ######################################################################
// ###                          ExcitedBaryons                        ###
// ######################################################################

class G4ExcitedBaryons : public G4VShortLivedParticle
{
 public:
   G4ExcitedBaryons(
       const G4String&     aName,        G4double            mass,
       G4double            width,        G4double            charge,   
       G4int               iSpin,        G4int               iParity,    
       G4int               iConjugation, G4int               iIsospin,   
       G4int               iIsospin3,    G4int               gParity,
       const G4String&     pType,        G4int               lepton,      
       G4int               baryon,       G4int               encoding,
       G4bool              stable,       G4double            lifetime,
       G4DecayTable        *decaytable
   );
   G4ExcitedBaryons*    ExcitedBaryonsDefinition(){return this;};
   G4ExcitedBaryons*    ExcitedBaryons(){return this;};
};


#endif






