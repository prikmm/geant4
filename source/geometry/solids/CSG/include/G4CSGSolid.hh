// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4CSGSolid.hh,v 2.0 1998/07/02 17:01:58 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
//  
// class G4CSGSolid
//
//  An abstract class for Constructed Solids.  Used primarily to 
// simplify inheritance tree.
//
// Member functions:
//
// As inherited from G4VSolid 
//
// History:
// 27.03.98 J.Apostolakis   Created first version.

#ifndef G4CSGSOLID_HH
#define G4CSGSOLID_HH

#include "G4VSolid.hh"

class G4CSGSolid : public G4VSolid {
public:
    G4CSGSolid(const G4String& pName);

    virtual ~G4CSGSolid();
};

#endif
