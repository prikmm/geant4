// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4DeuteronInelasticProcess.hh,v 2.0 1998/07/02 16:36:19 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
 // Hadronic Process: Deuteron Inelastic Process
 // J.L. Chuma, TRIUMF, 25-Feb-1997
 // Last modified: 03-Apr-1997

#ifndef G4DeuteronInelasticProcess_h
#define G4DeuteronInelasticProcess_h 1
 
//#include "G4HadronicInelasticProcess.hh"
#include "G4HadronInelasticProcess.hh"
 
// class G4DeuteronInelasticProcess : public G4HadronicInelasticProcess
 class G4DeuteronInelasticProcess : public G4HadronInelasticProcess
 {
 public:
    
    G4DeuteronInelasticProcess(
     const G4String& processName = "DeuteronInelastic" ) :
      //      G4HadronicInelasticProcess( processName, G4Deuteron::Deuteron() )
      G4HadronInelasticProcess( processName, G4Deuteron::Deuteron() )
    { }
    
    ~G4DeuteronInelasticProcess()
    { }
 };
 
#endif
 

