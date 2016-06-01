// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4AllocatorUnit.hh,v 2.0 1998/07/02 17:32:38 gunter Exp $
// GEANT4 tag $Name: geant4-00 $
//
// 
// ------------------------------------------------------------
//      GEANT 4 class header file 
//
//      For information related to this code contact:
//      CERN, CN Division, ASD group
//      History: first implementation, based on object model of
//      2nd December 1995, G.Cosmo
//      -------------- G4AllocatorUnit ----------------
//                by Tim Bell, September 1995
// ------------------------------------------------------------

#ifndef G4AllocatorUnit_h
#define G4AllocatorUnit_h 1

#include "globals.hh"

template <class Type>
class G4AllocatorUnit
{
public:
  int deleted;
  G4AllocatorUnit<Type> *fNext;
  Type fElement;
};

#endif
