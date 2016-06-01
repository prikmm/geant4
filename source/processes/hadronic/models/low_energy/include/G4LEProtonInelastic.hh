// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4LEProtonInelastic.hh,v 2.0 1998/07/02 16:28:36 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
 // Hadronic Process: Low Energy Proton Inelastic Process
 // original by H.P. Wellisch
 // modified by J.L. Chuma, TRIUMF, 19-Nov-1996
 // Last modified: 27-Mar-1997
 // Modified by J.L.Chuma 8-Jul-97 to implement Nucleus changes
 
#ifndef G4LEProtonInelastic_h
#define G4LEProtonInelastic_h 1
 
#include "G4InelasticInteraction.hh"
 
 class G4LEProtonInelastic : public G4InelasticInteraction
 {
 public:
    
    G4LEProtonInelastic() : G4InelasticInteraction()
    {
      SetMinEnergy( 0.0 );
      SetMaxEnergy( 25.*GeV );
    }
    
    ~G4LEProtonInelastic()
    { }
    
    G4VParticleChange *ApplyYourself( const G4Track &aTrack,
                                      G4Nucleus &targetNucleus );
 private:
    
    void Cascade(                               // derived from CASP
      G4FastVector<G4ReactionProduct,128> &vec,
      G4int &vecLen,
      const G4DynamicParticle *originalIncident,
      G4ReactionProduct &currentParticle,
      G4ReactionProduct &targetParticle,
      G4bool &incidentHasChanged, 
      G4bool &targetHasChanged,
      G4bool &quasiElastic );

    void SlowProton(
      const G4DynamicParticle *originalIncident,
      G4Nucleus &targetNucleus );
 };
 
#endif
 
