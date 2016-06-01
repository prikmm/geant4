// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4PhotoElectricEffect.cc,v 2.10 1998/11/17 10:59:19 maire Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// --------------------------------------------------------------
//      GEANT 4 class implementation file
//      CERN Geneva Switzerland
//
//      For information related to this code contact:
//      CERN, IT Division, ASD group
//      History: first implementation, based on object model of
//      2nd December 1995, G.Cosmo
//      ------------ G4PhotoElectricEffect physics process --------
//                   by Michel Maire, April 1996
// **************************************************************
// 12-06-96, Added SelectRandomAtom() method, by M.Maire
// 21-06-96, SetCuts implementation, M.Maire
// 17-09-96, PartialSumSigma(i)
//           split of ComputeBindingEnergy, M.Maire
// 08-01-97, crossection table + meanfreepath table, M.Maire
// 13-03-97, adapted for the new physics scheme, M.Maire
// 28-03-97, protection in BuildPhysicsTable, M.Maire
// 04-06-98, in DoIt, secondary production condition: range>min(threshold,safety)
// 13-08-98, new methods SetBining() PrintInfo()
// 17-11-98, use table of Atomic shells in PostStepDoIt
// --------------------------------------------------------------

#include "G4PhotoElectricEffect.hh"
#include "G4EnergyLossTables.hh"
#include "G4UnitsTable.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
// constructor
 
G4PhotoElectricEffect::G4PhotoElectricEffect(const G4String& processName)
  : G4VDiscreteProcess (processName),             // initialization
    theCrossSectionTable(NULL),
    theMeanFreePathTable(NULL),
    LowestEnergyLimit (10*keV),
    HighestEnergyLimit(50*MeV),
    NumbBinTable(100)
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
// destructor
 
G4PhotoElectricEffect::~G4PhotoElectricEffect()
{
   if (theCrossSectionTable) {
      theCrossSectionTable->clearAndDestroy();
      delete theCrossSectionTable;
   }

   if (theMeanFreePathTable) {
      theMeanFreePathTable->clearAndDestroy();
      delete theMeanFreePathTable;
   }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4PhotoElectricEffect::SetPhysicsTableBining(G4double lowE, G4double highE, G4int nBins)
{
  LowestEnergyLimit = lowE; HighestEnergyLimit = highE; NumbBinTable = nBins;
}  

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void G4PhotoElectricEffect::BuildPhysicsTable(const G4ParticleDefinition& PhotonType)

// Build microscopic cross section table and mean free path table
{
   G4double LowEdgeEnergy, Value;
   G4PhysicsLogVector* ptrVector;

// Build microscopic cross section tables for the Photo Electric Effect

   if (theCrossSectionTable) {
           theCrossSectionTable->clearAndDestroy(); delete theCrossSectionTable; }

   theCrossSectionTable = new G4PhysicsTable( G4Element::GetNumberOfElements()) ;
   const G4ElementTable* theElementTable = G4Element::GetElementTable() ;
   G4double AtomicNumber;
   G4int J;

   for ( J=0 ; J < G4Element::GetNumberOfElements(); J++ )  
      { 
        //create physics vector then fill it ....
        ptrVector = new G4PhysicsLogVector(LowestEnergyLimit, HighestEnergyLimit,
                                           NumbBinTable ) ;
        AtomicNumber = (*theElementTable)(J)->GetZ();
 
        for ( G4int i = 0 ; i < NumbBinTable ; i++ )      
           {
             LowEdgeEnergy = ptrVector->GetLowEdgeEnergy( i ) ;
             Value = ComputeCrossSectionPerAtom( LowEdgeEnergy, AtomicNumber);  
             ptrVector->PutValue( i , Value ) ;
           }

        theCrossSectionTable->insertAt( J , ptrVector ) ;

      }

// Build mean free path table for the Photo Electric Effect

   if (theMeanFreePathTable) {
           theMeanFreePathTable->clearAndDestroy(); delete theMeanFreePathTable; }

   theMeanFreePathTable = new G4PhysicsTable( G4Material::GetNumberOfMaterials() ) ;
   const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable() ;
   G4Material* material;

   for ( J=0 ; J < G4Material::GetNumberOfMaterials(); J++ )  
     { 
        //create physics vector then fill it ....
        ptrVector = new G4PhysicsLogVector(LowestEnergyLimit, HighestEnergyLimit,
                                           NumbBinTable ) ;
        material = (*theMaterialTable)(J);
 
        for ( G4int i = 0 ; i < NumbBinTable ; i++ )      
           {
             LowEdgeEnergy = ptrVector->GetLowEdgeEnergy( i ) ;
             Value = ComputeMeanFreePath( LowEdgeEnergy, material);  
             ptrVector->PutValue( i , Value ) ;
           }

        theMeanFreePathTable->insertAt( J , ptrVector ) ;

     }
                                    
    PrintInfoDefinition();  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
G4double G4PhotoElectricEffect::ComputeCrossSectionPerAtom (G4double PhotonEnergy,
                                                                G4double AtomicNumber)
 
// Calculates the microscopic cross section in GEANT4 internal units.
// A parametrized formula from L. Urban is used to estimate the total cross section.
// It gives a good description of the elements : 5 < Atomic Number < 100 and
//                                               from 10 keV to 50 MeV.
 
{
 G4double CrossSection = 0.0 ;
 if ( AtomicNumber < 1. )      return CrossSection;
 if ( PhotonEnergy > 50.*MeV ) return CrossSection;
      
 static const G4double
   p1K =-8.8893e+2*nanobarn, p2K = 2.4394   *nanobarn, p3K = 2.8835e+2*nanobarn,
   p4K = 1.2133e+1*nanobarn, p5K =-3.1104e+2*nanobarn, p6K =-1.7284e-1*nanobarn,
   p7K = 1.4400e+1*nanobarn, p8K = 6.8357e+1*nanobarn, p9K = 7.3945e-4*nanobarn,
   p10K=-4.8149e-2*nanobarn, p11K= 5.5823e-1*nanobarn, p12K=-1.0089e-1*nanobarn;
 static const G4double
   p1L1=-1.0927e+3*nanobarn, p2L1=-9.7897e-1*nanobarn, p3L1= 1.2854e+2*nanobarn;
 static const G4double
   p1L2=-4.5803e+3*nanobarn, p2L2= 1.6858e-3*nanobarn, p3L2= 1.2013e+2*nanobarn;
 static const G4double 
   p1M = 1.6924e+1*nanobarn;

 const G4double pwZ = 3.845 ,  pwE = 2.975 ; 
    
 G4double Z  = AtomicNumber,                  Z2  = Z*Z,   Z3  = Z*Z*Z;
 G4double Em = PhotonEnergy/electron_mass_c2, Em2 = Em*Em, Em3 = Em*Em*Em;

 CrossSection = pow(Z,pwZ)/pow(Em,pwE);

 if (PhotonEnergy > ComputeKBindingEnergy(Z) ) {
      CrossSection *= (p1K/Z  + p2K/Em + p3K + p4K*Z + p5K*Em
                    +  p6K*Z2 + p7K *Z *Em + p8K   *Em2
                    +  p9K*Z3 + p10K*Z2*Em + p11K*Z*Em2 + p12K*Em3);
      if (CrossSection < 0.) CrossSection = 0. ;
    }

 else if (PhotonEnergy > ComputeL1BindingEnergy(Z) ) {
      CrossSection *= (p1L1/Z + p2L1/Em + p3L1 );
      if (CrossSection < 0.) CrossSection = 0. ;
    }

 else if (PhotonEnergy > ComputeL2BindingEnergy(Z) ) {
      CrossSection *= (p1L2/Z + p2L2/Em + p3L2 );
      if (CrossSection < 0.) CrossSection = 0. ;
    }

 else CrossSection *= p1M;

 return CrossSection;
}
 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
G4VParticleChange* G4PhotoElectricEffect::PostStepDoIt(const G4Track& aTrack,
                                                      const G4Step&  aStep)
//
// Generate an electron resulting of a photo electric effect.
// The incident photon disappear.
// GEANT4 internal units
//
 
{
   aParticleChange.Initialize(aTrack);
   G4Material* aMaterial = aTrack.GetMaterial();

   const G4DynamicParticle* aDynamicPhoton = aTrack.GetDynamicParticle();

   G4double PhotonEnergy = aDynamicPhoton->GetKineticEnergy();
   G4ParticleMomentum PhotonDirection = aDynamicPhoton->GetMomentumDirection();

   
   // select randomly one element constituing the material.
   G4Element* anElement = SelectRandomAtom(aDynamicPhoton, aMaterial);

   //
   // Photo electron
   //

   G4int NbOfShells = anElement->GetNbOfAtomicShells();
   G4int i=0;
   while ((i<NbOfShells)&&(PhotonEnergy<anElement->GetAtomicShell(i))) i++;
   if (i==NbOfShells) return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
   
   G4double ElecKineEnergy = PhotonEnergy - anElement->GetAtomicShell(i);
   if ((G4EnergyLossTables::GetRange(G4Electron::Electron(),
        ElecKineEnergy,aMaterial)>aStep.GetPostStepPoint()->GetSafety())
        ||
       (ElecKineEnergy >
       (G4Electron::Electron()->GetCutsInEnergy())[aMaterial->GetIndex()]))
     {
      // the electron is created in the direction of the incident photon ...  
      G4DynamicParticle* aElectron= new G4DynamicParticle (G4Electron::Electron(),
                                                        PhotonDirection, ElecKineEnergy) ;
      aParticleChange.SetNumberOfSecondaries(1) ;
      aParticleChange.AddSecondary( aElectron ) ; 
     }
   else
     {
      ElecKineEnergy = 0. ;
      aParticleChange.SetNumberOfSecondaries(0) ;
     }

   //
   // Kill the incident photon 
   //

   aParticleChange.SetMomentumChange( 0., 0., 0. ) ;
   aParticleChange.SetEnergyChange( 0. ) ;
   aParticleChange.SetLocalEnergyDeposit( PhotonEnergy - ElecKineEnergy ) ;  
   aParticleChange.SetStatusChange( fStopAndKill ) ; 

   //  Reset NbOfInteractionLengthLeft and return aParticleChange
   return G4VDiscreteProcess::PostStepDoIt( aTrack, aStep );
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4Element*
G4PhotoElectricEffect::SelectRandomAtom(const G4DynamicParticle* aDynamicPhoton,
                                              G4Material* aMaterial)
{
  // select randomly 1 element within the material

  const G4int NumberOfElements            = aMaterial->GetNumberOfElements();
  const G4ElementVector* theElementVector = aMaterial->GetElementVector();
  if (NumberOfElements == 1) return (*theElementVector)(0);

  const G4double* NbOfAtomsPerVolume = aMaterial->GetVecNbOfAtomsPerVolume();

  G4double PartialSumSigma = 0. ;
  G4double rval = G4UniformRand()/MeanFreePath;
 
  for ( G4int elm=0 ; elm < NumberOfElements ; elm++ )
      { PartialSumSigma += NbOfAtomsPerVolume[elm] *
                   GetCrossSectionPerAtom(aDynamicPhoton,
                                          (*theElementVector)(elm));
        if (rval <= PartialSumSigma) return ((*theElementVector)(elm));
      }
  G4cout << " WARNING !!! - The Material '"<< aMaterial->GetName()
       << "' has no elements, NULL pointer returned." << endl;
  return NULL;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4PhotoElectricEffect::PrintInfoDefinition()
{
  G4String comments = "Total cross sections from a parametrisation(L.Urban). ";
           comments += "Good description from 10 KeV to 50 MeV for all Z";
                     
  G4cout << endl << GetProcessName() << ":  " << comments
         << "\n        PhysicsTables from " << G4BestUnit(LowestEnergyLimit,"Energy")
         << " to " << G4BestUnit(HighestEnergyLimit,"Energy") 
         << " in " << NumbBinTable << " bins. \n";
}         

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
