// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4OpenGLImmediateWin32.hh,v 2.0 1998/07/02 16:44:03 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// OpenGLImmediateWin32 graphics system factory.

#if defined (G4VIS_BUILD_OPENGLWIN32_DRIVER) || defined (G4VIS_USE_OPENGLWIN32)

#ifndef G4OPENGLIMMEDIATEWIN32_HH
#define G4OPENGLIMMEDIATEWIN32_HH

#include "G4VGraphicsSystem.hh"

class G4OpenGLImmediateWin32: public G4VGraphicsSystem {
public:
  G4OpenGLImmediateWin32 ();
  G4VScene* CreateScene ();
  G4VView*  CreateView  (G4VScene&);
};

#endif

#endif
