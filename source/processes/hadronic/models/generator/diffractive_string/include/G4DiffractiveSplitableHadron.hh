// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4DiffractiveSplitableHadron.hh,v 1.4 1998/11/05 17:55:38 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//

#ifndef G4DiffractiveSplitableHadron_h
#define G4DiffractiveSplitableHadron_h 1

// ------------------------------------------------------------
//      GEANT 4 class header file
//
//      For information related to this code contact:
//      CERN, CN Division, ASD group
//      ---------------- G4DiffractiveSplitableHadron----------------
//             by Gunter Folger, August 1998.
//       class splitting an interacting particle. Used by FTF String Model.
// ------------------------------------------------------------

#include "G4VSplitableHadron.hh"


class G4DiffractiveSplitableHadron : public G4VSplitableHadron
{

public:

	G4DiffractiveSplitableHadron(const G4ReactionProduct & aPrimary);
	G4DiffractiveSplitableHadron(const G4Nucleon & aNucleon);
	G4DiffractiveSplitableHadron(const G4VKineticNucleon * aNucleon);
	~G4DiffractiveSplitableHadron();


	int operator==(const G4DiffractiveSplitableHadron &right) const;
	int operator!=(const G4DiffractiveSplitableHadron &right) const;


	void SplitUp();
	G4Parton * GetNextParton() ;
	G4Parton * GetNextAntiParton();
	
private:
	G4DiffractiveSplitableHadron();
	G4DiffractiveSplitableHadron(const G4DiffractiveSplitableHadron &right);
	const G4DiffractiveSplitableHadron & operator=(const G4DiffractiveSplitableHadron &right);

//implementation
	G4int Diquark(G4int aquark,G4int bquark,G4int Spin) const; // to splitable hadron
	void ChooseStringEnds(G4int PDGcode,G4int * aEnd, G4int * bEnd) const; // to splitable hadron

private:
	G4Parton *Parton[2];
	G4int    PartonIndex; 

};

#endif	
