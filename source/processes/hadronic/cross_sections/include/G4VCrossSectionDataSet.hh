// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VCrossSectionDataSet.hh,v 2.1 1998/07/13 19:06:01 jwellisc Exp $
// GEANT4 tag $Name: geant4-00 $
//
//
// GEANT4 physics abstract class: G4VCrossSectionData -- header file
// F.W. Jones, TRIUMF, 20-JAN-97
//

#ifndef G4VCrossSectionDataSet_h
#define G4VCrossSectionDataSet_h 1

#include "G4DynamicParticle.hh"
#include "G4Element.hh"


class G4VCrossSectionDataSet
{
public:

   G4VCrossSectionDataSet() :
      verboseLevel(0)
   {
   }

   virtual ~G4VCrossSectionDataSet()
   {
   }

   virtual
   G4bool IsApplicable(const G4DynamicParticle*, const G4Element*) = 0;

   virtual
   G4double GetCrossSection(const G4DynamicParticle*, const G4Element*) = 0;

   virtual
   void BuildPhysicsTable(const G4ParticleDefinition&) = 0;

   virtual
   void DumpPhysicsTable(const G4ParticleDefinition&) = 0;

   void SetVerboseLevel(G4int value)
   {
      verboseLevel = value;
   }

   G4int GetVerboseLevel(G4int value)
   {
      return verboseLevel;
   }

protected:

   G4int verboseLevel;
};

#endif
