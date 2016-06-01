// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VContinuousDiscreteProcess.cc,v 2.0 1998/07/02 16:39:50 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// --------------------------------------------------------------
//	GEANT 4 class implementation file 
//
//	For information related to this code contact:
//	CERN, CN Division, ASD Group
//	History: first implementation, based on object model of
//	2nd December 1995, G.Cosmo
// --------------------------------------------------------------
//   New Physics scheme           8 Jan. 1997  H.Kurahige
// ------------------------------------------------------------

#include "G4VContinuousDiscreteProcess.hh"
G4VContinuousDiscreteProcess::G4VContinuousDiscreteProcess()
                   :G4VProcess("No Name Discrete Process") 
{
  G4Exception("G4VContinuousDiscreteProcess:: default constructor is called");
}

G4VContinuousDiscreteProcess::G4VContinuousDiscreteProcess(const G4String& aName , G4ProcessType aType)
                  : G4VProcess(aName, aType)
{
}

G4VContinuousDiscreteProcess::~G4VContinuousDiscreteProcess()
{
}

G4VContinuousDiscreteProcess::G4VContinuousDiscreteProcess(G4VContinuousDiscreteProcess& right)
                  : G4VProcess(right)
{
}











