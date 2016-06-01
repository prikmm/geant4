// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// -------------------------------------------------------------------
//      GEANT 4 class file 
//
//      For information related to this code contact:
//      CERN, IT Division, ASD group
//      CERN, Geneva, Switzerland
//
//      File name:     G4NuclearLevelManager
//
//      Author:        Maria Grazia Pia (pia@genova.infn.it)
// 
//      Creation date: 24 October 1998
//
//      Modifications: 
//      
// -------------------------------------------------------------------

#include "G4NuclearLevelManager.hh"

#include "globals.hh"
#include "G4NuclearLevel.hh"
#include "G4ios.hh"
#include <stdlib.h>
#include <fstream.h>
#include <strstream.h>

G4NuclearLevelManager::G4NuclearLevelManager():
  _A(0), _Z(0), _levels(0), _levelEnergy(0), _gammaEnergy(0), _probability(0)
{ }

G4NuclearLevelManager::G4NuclearLevelManager(G4int Z, G4int A): _Z(Z), _A(A)
{ 


  if (A <= 0 || Z <= 0 || Z > A )
    G4Exception("==== G4NuclearLevelManager ==== (Z,A) <0, or Z>A");

  _levels = 0;

  MakeLevels();
}


G4NuclearLevelManager::~G4NuclearLevelManager()
{ 
  if ( _levels ) {
    if (_levels->entries()>0) _levels->clearAndDestroy();
    delete _levels;
   _levels = 0;
  }
}

void G4NuclearLevelManager::SetNucleus(G4int Z, G4int A)
{
  if (_Z != Z || _A != A)
    {
      _A = A;
      _Z = Z;
      MakeLevels();
    }

}

G4bool G4NuclearLevelManager::IsValid(G4int Z, G4int A) const
{
  G4bool valid = true;

  if (A < 0 || Z < 0 || A < Z) valid = false;

  G4String dirName = getenv("G4LEVELGAMMADATA");
  char name[100] = {""};
  ostrstream ost(name, 100, ios::out);
  ost << dirName << "/" << "z" << Z << ".a" << A;
  G4String file(name); 

  ifstream inFile(file);
  if (! inFile) valid = false;  
  
  return valid;
}


G4int G4NuclearLevelManager::NumberOfLevels() const
{
  G4int n = 0;
  if (_levels != 0) n = _levels->entries();
  return n;
}


const G4PtrLevelVector* G4NuclearLevelManager::GetLevels() const
{
  return _levels;
}


const G4NuclearLevel* G4NuclearLevelManager::NearestLevel(G4double energy, G4double eDiffMax) const
{
  G4int iNear = -1;
  
  G4double diff = 9999. * GeV;
  if (_levels != 0)
    {
      G4int i = 0;
      for (i=0; i<_levels->entries(); i++)
	{
	  G4double e = _levels->at(i)->Energy();
	  G4double eDiff = abs(e - energy);
	  if (eDiff < diff && eDiff <= eDiffMax)
	    { 
	      diff = eDiff; 
	      iNear = i;
	    }
	}
    }
  if (_levels != 0 && iNear >= 0 && iNear < _levels->entries())
    { return _levels->at(iNear); }
  else
    { return 0; }
}


G4double G4NuclearLevelManager::MinLevelEnergy() const
{
  G4double eMin = 9999.*GeV;
  if (_levels != 0)
    {
      if (_levels->entries() > 0) eMin = _levels->first()->Energy(); 
    }
  return eMin;
}


G4double G4NuclearLevelManager::MaxLevelEnergy() const
{
  G4double eMax = 0.;
  if (_levels != 0)
    {
      if (_levels->entries() > 0) eMax = _levels->last()->Energy(); 
    }
  return eMax;
}


const G4NuclearLevel* G4NuclearLevelManager::HighestLevel() const
{
  if (_levels!= 0 && _levels->entries() > 0) return _levels->first(); 
  else return 0; 
}


const G4NuclearLevel* G4NuclearLevelManager::LowestLevel() const
{
  if (_levels != 0 && _levels->entries() > 0) return _levels->last();
  else return 0;
}


G4bool G4NuclearLevelManager::Read(ifstream& dataFile)
{
  const G4double minProbability = 0.001;

  G4bool result = true;

  if (dataFile >> _levelEnergy)
    {
      dataFile >> _gammaEnergy >> _probability;
      _levelEnergy *= keV;
      _gammaEnergy *= keV;

      // The following adjustment is needed to take care of anomalies in 
      // data files, where some transitions show up with relative probability
      // zero
      if (_probability < minProbability) _probability = minProbability;

      // G4cout << "Read " << _levelEnergy << " " << _gammaEnergy << " " << _probability << endl;
    }
  else
    {
      result = false;
    }

  return result;
}


void G4NuclearLevelManager::MakeLevels()
{
  G4String dirName = getenv("G4LEVELGAMMADATA");
  char name[100] = {""};
  ostrstream ost(name, 100, ios::out);
  ost << dirName << "/" << "z" << _Z << ".a" << _A;
  G4String file(name); 

  ifstream inFile(file, ios::in);
  
  if (! inFile) 
    {
      //      G4cout << " G4NuclearLevelManager: (" << _Z << "," << _A 
      //  	     << ") does not have LevelsAndGammas file" << endl;
      return;
    }

  if (_levels != 0)
    {
      if (_levels->entries()>0) _levels->clearAndDestroy();
      delete _levels;
    }

  _levels = new G4PtrLevelVector;

  G4DataVector eLevel;
  G4DataVector eGamma;
  G4DataVector wGamma;

  while (Read(inFile))
    {
      eLevel.insert(_levelEnergy);
      eGamma.insert(_gammaEnergy);
      wGamma.insert(_probability);
    }

  // ---- MGP ---- Don't forget to close the file 
  inFile.close();

  G4int nData = eLevel.entries();

  //  G4cout << " ==== MakeLevels ===== " << nData << " data read " << endl;

  G4double thisLevelEnergy = eLevel.at(0);
  G4DataVector thisLevelEnergies;
  G4DataVector thisLevelWeights;

  G4double e = -1.;
  G4int i;
  for (i=0; i<nData; i++)
    {
      e = eLevel.at(i);
      if (e != thisLevelEnergy)
	{
	  //	  G4cout << "Making a new level... " << e << " " 
	  //		 << thisLevelEnergies.entries() << " " 
	  //		 << thisLevelWeights.entries() << endl;

	  G4NuclearLevel* newLevel = new G4NuclearLevel(thisLevelEnergy,thisLevelEnergies,thisLevelWeights);
	  _levels->insert(newLevel);
	  // Reset data vectors
	  thisLevelEnergies.clear();
	  thisLevelWeights.clear();
	  thisLevelEnergy = e;
	}
      // Append current data
      thisLevelEnergies.insert(eGamma.at(i));
      thisLevelWeights.insert(wGamma.at(i));
    }
  // Make last level
  if (e > 0.)
    {
      G4NuclearLevel* newLevel = new G4NuclearLevel(e,thisLevelEnergies,thisLevelWeights);
      _levels->insert(newLevel);
    }

  return;
}


void G4NuclearLevelManager::PrintAll()
{
  G4int nLevels = 0;
  if (_levels != 0) nLevels = _levels->entries();

  G4cout << " ==== G4NuclearLevelManager ==== (" << _Z << ", " << _A << ") has " 
	 << nLevels << " levels" << endl
	 << "Highest level is at energy " << MaxLevelEnergy() << " MeV " << endl
	 << "Lowest level is at energy " << MinLevelEnergy() << " MeV " << endl;

  G4int i = 0;
  for (i=0; i<nLevels; i++)
    { _levels->at(i)->PrintAll(); }
}


G4NuclearLevelManager::G4NuclearLevelManager(const G4NuclearLevelManager &right)
{
  _levelEnergy = right._levelEnergy;
  _gammaEnergy = right._gammaEnergy;
  _probability = right._probability;
  _A = right._A;
  _Z = right._Z;
  if (right._levels != 0)   
    {
      _levels = new G4PtrLevelVector;
      G4int n = right._levels->entries();
      G4int i;
      for (i=0; i<n; i++)
	{
	  _levels->insert(new G4NuclearLevel(*(right._levels->at(i))));
	}
    }
  else 
    {
      _levels = 0;
    }
}
