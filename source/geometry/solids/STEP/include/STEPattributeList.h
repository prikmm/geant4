// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: STEPattributeList.h,v 2.1 1998/07/12 02:57:15 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//

#ifndef _STEPattributeList_h
#define _STEPattributeList_h 1

/*
* NIST STEP Core Class Library
* clstepcore/STEPattributeList.h
* May 1995
* K. C. Morris
* David Sauder

* Development of this software was funded by the United States Government,
* and is not subject to copyright.
*/

/*  */

#ifdef __O3DB__
#include <OpenOODB.h>
#endif

//#ifndef _STEPattribute_typedefs
//#define _STEPattribute_typedefs 1

#include <STEPattribute.h>
#include <SingleLinkList.h>

//class STEPattribute;
class STEPattributeList;
//class AttrListNode;

class AttrListNode :  public SingleLinkNode 
{
  friend class STEPattributeList;

  protected:
    STEPattribute *attr;

  public:
    AttrListNode(STEPattribute *a) { attr = a; }
};

class STEPattributeList : public SingleLinkList
{
  public:
    STEPattributeList() { }

    STEPattribute& operator [] (int n);
    int list_length();
    void push(STEPattribute *a);

};

/*****************************************************************
**                                                              **
**      This file defines the type STEPattributeList -- a List  **
**      of pointers to STEPattribute objects.  The nodes on the **
**      List point to STEPattributes.  
**                                                              **
		USED TO BE - DAS
**      The file was generated by using GNU's genclass.sh       **
**      script with the List prototype definitions.  The        **
**      command to Generate it was as follows:                  **

        genclass.sh STEPattribute ref List STEPattribute

**      The file is dependent on the file "STEPattribute.h"     **
**      which contains the definition of STEPattribute.         **
**                                                              **
**      1/15/91  kcm                                            **
*****************************************************************/

#endif
