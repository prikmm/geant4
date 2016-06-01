#ifndef G4VElasticScatterer_h
#define G4VElasticScatterer_h 1

#include "G4KineticTrackVector.hh"

class G4VElasticScatterer 
   {
public:
    G4VElasticScatterer();
   ~G4VElasticScatterer();
public:
    virtual G4KineticTrackVector* Scatter(const G4KineticTrack &aProjectile, const G4KineticTrack &aTarget)= 0;
};
  
#endif 


