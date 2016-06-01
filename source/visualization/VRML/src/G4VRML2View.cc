// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VRML2View.cc,v 2.2 1998/11/09 19:33:40 allison Exp $
// GEANT4 tag $Name: geant4-00 $
//
// G4VRML2View.cc
// Satoshi Tanaka & Yasuhide Sawada

//=================//
#ifdef G4VIS_BUILD_VRML_DRIVER
//=================//


//#define DEBUG_FR_VIEW

#include <math.h>

#include "G4SceneData.hh"
#include "G4VRML2View.hh"
#include "G4VRML2Scene.hh"
#include "G4VRML2.hh"
#include "G4ios.hh"

G4VRML2View::G4VRML2View(G4VRML2Scene& scene, const G4String& name) :
 G4VView(scene, scene.IncrementViewCount(), name),
 fScene(scene),
 fDest(scene.fDest)
{
	fViewHalfAngle = 0.5 * 0.785398 ; // 0.5 * 45*deg
	fsin_VHA       = sin ( fViewHalfAngle ) ;	
}

G4VRML2View::~G4VRML2View()
{}

void G4VRML2View::SetView()
{
#if defined DEBUG_FR_VIEW
	G4cerr << "***** G4VRML2View::SetView()" << endl;
	G4cerr << "G4VRML2View::SetView(); not imlemented. " << endl;
#endif

// Do nothing, since VRML a browser is running as a different process.
// SendViewParameters () will do this job instead.

}


void G4VRML2View::DrawView()
{
#if defined DEBUG_FR_VIEW
	G4cerr << "***** G4VRML2View::DrawView()" << endl;
#endif
	// Open VRML2 file and output header comments
	fScene.beginSending() ;

        // Viewpoint node
        SendViewParameters(); 

	// Here is a minimal DrawView() function.
	NeedKernelVisit();
	ProcessView();
	FinishView();
}

void G4VRML2View::ClearView(void)
{
#if defined DEBUG_FR_VIEW
	G4cerr << "***** G4VRML2View::ClearView()" << endl;
	G4cerr << "G4VRML2View::ClearView(); not implemented. " << endl;
#endif
}

void G4VRML2View::ShowView(void)
{
#if defined DEBUG_FR_VIEW
	G4cerr << "***** G4VRML2View::ShowView()" << endl;
#endif
	fScene.endSending();
}

void G4VRML2View::FinishView(void)
{
#if defined DEBUG_FR_VIEW
	G4cerr << "***** G4VRML2View::FinishView()" << endl;
	//G4cerr << "G4VRML2View::FinishView(); not implemented. " << endl;
#endif
	//fScene.endSending();
}

void G4VRML2View::SendViewParameters () 
{
  // Calculates view representation based on extent of object being
  // viewed and (initial) direction of camera.  (Note: it can change
  // later due to user interaction via visualization system's GUI.)

#if defined DEBUG_FR_VIEW
      G4cerr << "***** G4VRML2View::SendViewParameters()\n";
#endif 
	// error recovery
	if ( fsin_VHA < 1.0e-6 ) { return ; } 

	// camera distance
	G4double extent_radius = fScene.GetSceneData().GetExtent().GetExtentRadius();
	G4double camera_distance = extent_radius / fsin_VHA ;

	// camera position on Z axis
	const G4Point3D&	target_point = fVP.GetCurrentTargetPoint();
	G4double		E_z = target_point.z() + camera_distance;
	G4Point3D		E(0.0, 0.0, E_z );

	// VRML codes are generated below	
	fDest << endl;
	fDest << "#---------- CAMERA" << endl;
	fDest << "Viewpoint {"         << endl;
	fDest << "\t" << "position "           ;
	fDest                 << E.x() << " "  ;
	fDest                 << E.y() << " "  ;
	fDest                 << E.z() << endl ;
	fDest << "}" << endl;
	fDest << endl;

} 



#endif
