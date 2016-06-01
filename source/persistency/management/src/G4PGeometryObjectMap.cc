// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4PGeometryObjectMap.cc,v 2.2 1998/07/14 14:15:04 morita Exp $
// GEANT4 tag $Name: geant4-00 $
//
// class G4PGeometryObjectMap 
//
// Implementation of a class responsible for keeping track of
// the geometry object map.
//
// History:
// 98.06.20 Y.Morita  Initial version

#include "G4PGeometryObjectMap.hh"

// tell C++ compiler to generate code of parametrized class for eacy class
// (Required by Objectivity/DB)
implement(HepRefVArray,G4PVPhysicalVolume)
implement(HepRefVArray,G4PLogicalVolume)
implement(HepRefVArray,G4PVSolid)

G4PGeometryObjectMap::G4PGeometryObjectMap()
{
}

G4PGeometryObjectMap::G4PGeometryObjectMap( const G4String theGeometryName )
{
  noPhysVol = 0;
  noLogVol  = 0;
  noSolids  = 0;
}

G4PGeometryObjectMap::~G4PGeometryObjectMap()
{
}

// -------- LookUp() and Add() functions for each class -------- //

HepRef(G4PVPhysicalVolume) G4PGeometryObjectMap::LookUp(
                                      G4VPhysicalVolume* inGeomObj )
{
  assert(inGeomObj != NULL);
  for(G4int i=0;i<noPhysVol;i++)
    if( transPhysVolPtrs[i] == inGeomObj )
      return persPhysVolPtrs[i];

  return NULL;
}

void G4PGeometryObjectMap::Add(         G4VPhysicalVolume*  inGeomObj,
                                HepRef(G4PVPhysicalVolume) outGeomObj )
{
  assert(inGeomObj != NULL);
  HepRef(G4PVPhysicalVolume) aGeomObj = LookUp( inGeomObj );
  if( aGeomObj == NULL )
  {
//    assert( inGeomObj == aGeomObj );
    noPhysVol++;
    transPhysVolPtrs.resize(noPhysVol);
    transPhysVolPtrs[noPhysVol-1] = inGeomObj;
     persPhysVolPtrs.append(outGeomObj);
  }
}

// ------------------------------------------------------------- //

G4VPhysicalVolume* G4PGeometryObjectMap::LookUp(
                               HepRef(G4PVPhysicalVolume) inGeomObj )
{
  assert(inGeomObj != NULL);
  for(G4int i=0;i<noPhysVol;i++)
    if( persPhysVolPtrs[i] == inGeomObj )
      return transPhysVolPtrs[i];

  return NULL;
}

void G4PGeometryObjectMap::Add( HepRef(G4PVPhysicalVolume)  inGeomObj,
                                        G4VPhysicalVolume* outGeomObj )
{
  assert(inGeomObj != NULL);
  for(G4int i=0;i<noPhysVol;i++)
  {
    if( persPhysVolPtrs[i] == inGeomObj )
    {
      transPhysVolPtrs[i] = outGeomObj;
      break;
    }
    cerr << "G4PGeometryObjectMap::Add -- transient solid not assigned" << endl;
  }
}

// ------------------------------------------------------------- //

HepRef(G4PLogicalVolume) G4PGeometryObjectMap::LookUp(
                                      G4LogicalVolume* inGeomObj )
{
  assert(inGeomObj != NULL);
  for(G4int i=0;i<noLogVol;i++)
    if( transLogVolPtrs[i] == inGeomObj )
      return persLogVolPtrs[i];

  return NULL;
}

void G4PGeometryObjectMap::Add(         G4LogicalVolume*  inGeomObj,
                                HepRef(G4PLogicalVolume) outGeomObj )
{
  assert(inGeomObj != NULL);
  HepRef(G4PLogicalVolume) aGeomObj = LookUp( inGeomObj );
  if( aGeomObj == NULL )
  {
//    assert( inGeomObj == aGeomObj );
    noLogVol++;
    transLogVolPtrs.resize(noLogVol);
    transLogVolPtrs[noLogVol-1] = inGeomObj;
     persLogVolPtrs.append(outGeomObj);
  }
}

// ------------------------------------------------------------- //

G4LogicalVolume* G4PGeometryObjectMap::LookUp(
                               HepRef(G4PLogicalVolume) inGeomObj )
{
  assert(inGeomObj != NULL);
  for(G4int i=0;i<noLogVol;i++)
    if( persLogVolPtrs[i] == inGeomObj )
      return transLogVolPtrs[i];

  return NULL;
}

void G4PGeometryObjectMap::Add( HepRef(G4PLogicalVolume)  inGeomObj,
                                        G4LogicalVolume* outGeomObj )
{
  assert(inGeomObj != NULL);
  for(G4int i=0;i<noLogVol;i++)
  {
    if( persLogVolPtrs[i] == inGeomObj )
    {
      transLogVolPtrs[i] = outGeomObj;
      break;
    }
    cerr << "G4PGeometryObjectMap::Add -- transient solid not assigned" << endl;
  }
}

// ------------------------------------------------------------- //
HepRef(G4PVSolid) G4PGeometryObjectMap::LookUp( G4VSolid* inGeomObj )
{

#ifdef G4PERSISTENCY_DEBUG
  cout << "G4PGeometryObjectMap::LookUp(G4VSolid) -- noSolids is "
       << noSolids << endl;
  cout << "  G4VSolid info: " << inGeomObj << endl;
#endif

  assert(inGeomObj != NULL);
  for(G4int i=0;i<noSolids;i++)
  {
#ifdef G4PERSISTENCY_DEBUG
    G4VSolid* tmpSolid = transSolidPtrs[i];
    cout << "[" << i << "] transSolidPtrs[i]=" << tmpSolid <<endl;
    cout             << "   persSolidPtrs[i]=";
    HepRef(G4PVSolid) tmpPSolid = persSolidPtrs[i];
    tmpPSolid.print();
#endif

    if( transSolidPtrs[i] == inGeomObj )
      return persSolidPtrs[i];
  }

  return NULL;
}

void G4PGeometryObjectMap::Add(         G4VSolid*  inGeomObj,
                                HepRef(G4PVSolid) outGeomObj )
{
  assert(inGeomObj != NULL);
  HepRef(G4PVSolid) aGeomObj = LookUp( inGeomObj );
  if( aGeomObj == NULL )
  {
//    assert( inGeomObj == aGeomObj );
    noSolids++;
    transSolidPtrs.resize(noSolids);
    transSolidPtrs[noSolids-1] = inGeomObj;
     persSolidPtrs.append(outGeomObj);
  }
}

// ------------------------------------------------------------- //

G4VSolid* G4PGeometryObjectMap::LookUp(
                               HepRef(G4PVSolid) inGeomObj )
{

#ifdef G4PERSISTENCY_DEBUG
  cout << "G4PGeometryObjectMap::LookUp(G4PVSolid) -- noSolids is "
       << noSolids << endl;
  cout << "  G4PVSolid info:";
  inGeomObj.print(stdout);
#endif

  assert(inGeomObj != NULL);
  for(G4int i=0;i<noSolids;i++)
  {
#ifdef G4PERSISTENCY_DEBUG
    cout << "[" << i << "]  persSolidPtrs[i]=";
    HepRef(G4PVSolid) tmpPSolid = persSolidPtrs[i];
    tmpPSolid.print();
    G4VSolid* tmpSolid = transSolidPtrs[i];
    cout              << " transSolidPtrs[i]=" << tmpSolid <<endl;
#endif
    if( persSolidPtrs[i] == inGeomObj )
      return transSolidPtrs[i];
  }

  return NULL;
}

void G4PGeometryObjectMap::Add( HepRef(G4PVSolid)  inGeomObj,
                                        G4VSolid* outGeomObj )
{
  assert(inGeomObj != NULL);
  for(G4int i=0;i<noSolids;i++)
  {
    if( persSolidPtrs[i] == inGeomObj )
    {
      transSolidPtrs[i] = outGeomObj;
      break;
    }
    cerr << "G4PGeometryObjectMap::Add -- transient solid not assigned" << endl;
  }
}

// ------------------------------------------------------------- //

G4VSolid* G4PGeometryObjectMap::GetSolid(G4int i)
{
  if( i >= 0 && i < noSolids)
    return transSolidPtrs[i];
  else
    return NULL;
}

HepRef(G4PVSolid) G4PGeometryObjectMap::GetPSolid(G4int i)
{
  if( i >= 0 && i < noSolids)
    return persSolidPtrs[i];
  else
    return NULL;
}

void G4PGeometryObjectMap::InitTransientMap()
{
  transPhysVolPtrs.resize(noPhysVol);
  transLogVolPtrs.resize(noLogVol);
  transSolidPtrs.resize(noSolids);

#ifdef G4PERSISTENCY_DEBUG
  cout << "G4PGeometryObjectMap::InitTransientMap()" << endl;
  cout << " -- noPhysVol: " << noPhysVol << endl;
  cout << " -- transPhysVolPtrs.length: " << transPhysVolPtrs.length() << endl;
  cout << " -- transLogVolPtrs.length: "  << transLogVolPtrs.length()  << endl;
  cout << " -- transSolidPtrs.length: "   << transSolidPtrs.length()   << endl;
#endif

  for(G4int i=0;i<noPhysVol;i++)
    transPhysVolPtrs[i] = NULL;
  for(i=0;i<noLogVol;i++)
    transLogVolPtrs[i] = NULL;
  for(i=0;i<noSolids;i++)
    transSolidPtrs[i] = NULL;
}
