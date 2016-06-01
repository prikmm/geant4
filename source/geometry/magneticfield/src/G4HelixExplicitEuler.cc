// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4HelixExplicitEuler.cc,v 2.3 1998/11/13 14:30:21 japost Exp $
// GEANT4 tag $Name: geant4-00 $
//
#include "G4HelixExplicitEuler.hh"
#include "G4ThreeVector.hh"

//
//  Helix Explicit Euler: x_1 = x_0 + helix(h)
//  with helix(h) being a helix piece of length h
//  W.Wander <wwc@mit.edu> 12/09/97 
//

// -------------------------------------------------------------------------

// most simple approach for solving linear differential equations.
// Take the current derivative and add it to the current position.
//

void
G4HelixExplicitEuler::DumbStepper( const G4double  yIn[],
			      const G4double  dydx[],
			      const G4double  h,
			 	    G4double  yOut[])
{
  AdvanceHelix(yIn, dydx, h, yOut);

  // NormaliseTangentVector( yOut );  // this could harm more than it helps 
  return ;
}  
