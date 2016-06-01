// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VRML1Scene.cc,v 2.4 1998/11/09 19:33:24 allison Exp $
// GEANT4 tag $Name: geant4-00 $
//
// G4VRML1Scene.cc
// Yasuhide Sawada and Satoshi Tanaka

//=================//
#ifdef G4VIS_BUILD_VRML_DRIVER
//=================//


//#define DEBUG_FR_SCENE

#include <unistd.h>
#include <fstream.h>

#include "globals.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Point3D.hh"
#include "G4VisAttributes.hh"
#include "G4Transform.hh"
#include "G4Polyhedron.hh"
#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Polyline.hh"
#include "G4Trd.hh"
#include "G4Tubs.hh"
#include "G4Text.hh"
#include "G4Circle.hh"
#include "G4Square.hh"

#include "G4VRML1Scene.hh"
#include "G4VRML1View.hh"
#include "G4VRML1.hh"



G4VRML1Scene::G4VRML1Scene(G4VRML1& system, const G4String& name) :
	G4VScene(system, fSceneIdCount++, name),
	fSystem(system),
	fDest()
{
	fSceneCount++;
	fCurrentDEF = "";
}


G4VRML1Scene::~G4VRML1Scene()
{
#if defined DEBUG_FR_SCENE
	G4cerr << "***** ~G4VRML1Scene" << endl;
#endif 
	fSceneCount--;
}



#define  G4VRML1SCENE  G4VRML1Scene
#define  IS_CONNECTED  fDest.isConnected() 
#include "G4VRML1SceneFunc.icc"
#undef   IS_CONNECTED
#undef   G4VRML1SCENE 


void G4VRML1Scene::connectPort(G4int max_trial)
{
	G4int trial = 0 ;
	int port = fSystem.getPort();
	for (trial = 0; !fDest.isConnected()&& trial < max_trial; trial++, port++ ) {
		if (fDest.connect( (const char * )fSystem.getHostName(), port)) {
		    // INET domain connection is established
			G4cerr << "*** GEANT4 is connected to port  ";
			G4cerr << fDest.getPort(); 
			G4cerr << " of server  " << fSystem.getHostName() << endl;
			break; 
		} else { 
			// Connection failed. Try the next port.
			G4cerr << "*** GEANT4 incremented targeting port to ";
			G4cerr << port << endl;
		}

		sleep (1);

	} // for

	if (!fDest.isConnected()) {
		G4cerr << "*** INET Connection failed. " << endl;
		G4cerr << "    Maybe, you have not invoked viewer  g4vrmlview  yet, " << endl;
		G4cerr << "    or too many viewers are already running in the " << endl;
		G4cerr << "    server host(" << fSystem.getHostName() << "). " << endl;
	}
}

void G4VRML1Scene::closePort()
{
	fDest.close();
	G4cerr << "*** INET Connection closed. " << endl;
}


G4int G4VRML1Scene::fSceneIdCount = 0;
G4int G4VRML1Scene::fSceneCount = 0;

#endif
