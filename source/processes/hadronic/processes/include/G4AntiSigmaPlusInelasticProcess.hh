// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4AntiSigmaPlusInelasticProcess.hh,v 2.0 1998/07/02 16:36:15 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
 // Hadronic Process: AntiSigmaPlus Inelastic Process
 // J.L. Chuma, TRIUMF, 18-Feb-1997
 // Last modified: 03-Apr-1997
 
 // Note:  there is no .cc file
 
#ifndef G4AntiSigmaPlusInelasticProcess_h
#define G4AntiSigmaPlusInelasticProcess_h 1
 
//#include "G4HadronicInelasticProcess.hh"
#include "G4HadronInelasticProcess.hh"
 
// class G4AntiSigmaPlusInelasticProcess : public G4HadronicInelasticProcess
 class G4AntiSigmaPlusInelasticProcess : public G4HadronInelasticProcess
 {
 public:
    
    G4AntiSigmaPlusInelasticProcess(
     const G4String& processName = "AntiSigmaPlusInelastic" ) :
      //      G4HadronicInelasticProcess( processName, G4AntiSigmaPlus::AntiSigmaPlus() )
      G4HadronInelasticProcess( processName, G4AntiSigmaPlus::AntiSigmaPlus() )
    { }
    
    ~G4AntiSigmaPlusInelasticProcess()
    { }
 };
 
#endif
 

