// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4OpenGLImmediateView.cc,v 2.1 1998/07/13 17:11:25 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// Andrew Walkden  7th February 1997
// Class G4OpenGLImmediateView : Encapsulates the `immediateness' of
//                               an OpenGL view, for inheritance by
//                               derived (X, Xm...) classes.

#ifdef G4VIS_BUILD_OPENGL_DRIVER

#include "G4OpenGLImmediateView.hh"

#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>

#include "G4ios.hh"
#include <assert.h>
#include <unistd.h>

G4OpenGLImmediateView::G4OpenGLImmediateView (G4OpenGLImmediateScene& scene):
G4VView (scene, -1),
G4OpenGLView (scene),
fScene (scene)
{}

#endif
