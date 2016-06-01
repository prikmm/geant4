// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4OpenGLStoredX.hh,v 2.1 1998/11/06 13:42:20 allison Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// Andrew Walkden  10th February 1997
// OpenGL graphics system factory.

#if defined (G4VIS_BUILD_OPENGLX_DRIVER) || defined (G4VIS_USE_OPENGLX)

#ifndef G4OPENGLSTOREDX_HH
#define G4OPENGLSTOREDX_HH

#include "G4VGraphicsSystem.hh"

class G4OpenGLStoredX: public G4VGraphicsSystem {
public:
  G4OpenGLStoredX ();
  G4VScene* CreateScene (const G4String& name = "");
  G4VView*  CreateView  (G4VScene&, const G4String& name = "");
};

#endif

#endif
