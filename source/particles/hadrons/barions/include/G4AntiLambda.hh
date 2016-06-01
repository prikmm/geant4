// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4AntiLambda.hh,v 2.3 1998/09/24 06:48:40 kurasige Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// ------------------------------------------------------------
//      GEANT 4 class header file
//
//      For information related to this code contact:
//      CERN, CN Division, ASD group
//      History: first implementation, based on object model of
//      4-th April 1996, G.Cosmo
// ****************************************************************
//  Added particle definitions, H.Kurashige, 14 Feb 19
// ----------------------------------------------------------------

// Each class inheriting from G4VBarion
// corresponds to a particle type; one and only one
// instance for each class is guaranteed.

#ifndef G4AntiLambda_h
#define G4AntiLambda_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "G4VBarion.hh"

// ######################################################################
// ###                          AntiLambda                            ###
// ######################################################################

class G4AntiLambda : public G4VBarion
{
 private:
   static G4AntiLambda theAntiLambda;
   static G4double  theAntiLambdaLengthCut;
   static G4double* theAntiLambdaKineticEnergyCuts;

 private:
   G4AntiLambda(
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

 public:
   static G4AntiLambda* AntiLambdaDefinition();
   static G4AntiLambda* AntiLambda(){return &theAntiLambda;}
   static G4double GetCuts() {return theAntiLambdaLengthCut;}   
   static G4double* GetCutsInEnergy() {return theAntiLambdaKineticEnergyCuts;};

   void SetCuts(G4double aCut); 
};

#endif



