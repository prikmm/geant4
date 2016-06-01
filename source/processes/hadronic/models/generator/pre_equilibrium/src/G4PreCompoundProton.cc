// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
//
// by V. Lara
// Correction by V. Krylov

#include "G4PreCompoundProton.hh"



G4double G4PreCompoundProton::ProbabilityDistributionFunction(const G4double & eKin,
							      const G4Fragment & aFragment)
{
  const G4double r0 = 1.5; // fm
  const G4double SingleParticleLevelDensity = 
    0.595*G4PreCompoundParameters::GetAddress()->GetLevelDensity(); // AC
  G4double R0J=1.2;
  G4double C1 = eKin - GetCoulombBarrier();

  return 0.000234*r0*r0*pow(GetRestA(),2.0/3.0)*R0J*       
           GetExcitonLevelDensityRatio()/
    (SingleParticleLevelDensity*(aFragment.GetExcitationEnergy()/MeV)*GetRestA())*
    pow((1.0 - (eKin+GetBindingEnergy())/(aFragment.GetExcitationEnergy()/MeV)),
	(aFragment.GetNumberOfExcitons()-2.0))*C1;

  // Corrected some mistakes in return statement by V. Krylov:
  //     - First GetRestA() was GetA()
  //     - The C1 factor was inside of precedent pow(  )
}




G4double G4PreCompoundProton::GetKineticEnergy(const G4Fragment & aFragment)
{
  G4double DJ = - GetCoulombBarrier();

  G4double T = aFragment.GetNumberOfParticles() + aFragment.GetNumberOfHoles() - GetA() - 1.0;
  G4double R2 = GetMaximalKineticEnergy();
  G4double R1 = R2 + GetCoulombBarrier();
	

  if (T <= -0.1) return R1; 
  else if (T <= 0.1)  return sqrt(G4UniformRand())*R2 + GetCoulombBarrier();
  else {
    G4double E1 = (R1 - DJ*T)/(T + 1.0);
    G4double E = 0.0;
    G4double T3 = 0.0;
    do {
      E = GetCoulombBarrier()+G4UniformRand()*R2;
      G4double T1 = (E + DJ)/(E1 + DJ);
      G4double T2 = (R1 - E)/(R1 - E1);
      T3 = T1*pow(T2,T);
    } while (G4UniformRand() > T3);
    return E;
  }
}
