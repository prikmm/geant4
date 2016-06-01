// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4OpenGLXmStyleCallbacks.hh,v 2.0 1998/07/02 16:45:22 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
//

#ifdef G4VIS_BUILD_OPENGLXM_DRIVER

#ifndef G4OPENGLXMSTYLECALLBACKS_HH
#define G4OPENGLXMSTYLECALLBACKS_HH

#include "G4OpenGLXmView.hh"

void drawing_style_callback (Widget w, 
			     XtPointer clientData, 
			     XtPointer callData);

void rep_style_callback (Widget w, 
			 XtPointer clientData, 
			 XtPointer callData);

void background_color_callback (Widget w, 
				XtPointer clientData, 
				XtPointer callData);

void projection_callback (Widget w, 
			  XtPointer clientData, 
			  XtPointer callData);

#endif

#endif
