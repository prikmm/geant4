// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4PointRat.cc,v 2.2 1998/10/20 16:33:56 broglia Exp $
// GEANT4 tag $Name: geant4-00 $
//
//
// Modif 8 oct 98 : A.Floquet
//      G4PointRat datas are made of
// 	     . a point 3D
//	     . a additional value : the scale factor which is set to 1 by default
//

#include "G4PointRat.hh"

G4PointRat::G4PointRat():pt3d(){s=1;}

G4PointRat::G4PointRat(const G4Point3D& tmp):pt3d(tmp){s=1;}

G4PointRat::~G4PointRat(){}

void G4PointRat::operator=(const G4PointRat& a)
{   pt3d.setX(a.x());
    pt3d.setY(a.y());
    pt3d.setZ(a.z());
    s=a.w();
}

void G4PointRat::operator=(const G4Point3D& a)
{   pt3d = a;
    s=1;
}

void G4PointRat::CopyRationalValue(const RealNode& RNode)
{
  s = RNode.value;
}



