// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: STEPattributeList.cc,v 2.1 1998/07/13 16:53:49 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//

/*
* NIST STEP Core Class Library
* clstepcore/STEPattributeList.cc
* May 1995
* K. C. Morris
* David Sauder

* Development of this software was funded by the United States Government,
* and is not subject to copyright.
*/

/*  */

#include <STEPattributeList.h>
#include <STEPattribute.h>

//#include <stdlib.h>


STEPattribute& STEPattributeList::operator [] (int n)
{
    int x = 0;
    AttrListNode* a = (AttrListNode *)head;
    int cnt =  EntryCount();
    if (n < cnt)
	while (a && (x < n))
	{
	    a = (AttrListNode *)(a->next);
	    x++;
	}
    if (a)  return *(a->attr);

   // else
    G4cerr << "\nERROR in STEP Core library:  " << __FILE__ <<  ":"
      << __LINE__ << "\n" << _POC_ << "\n\n";
    return *(((AttrListNode *)head)->attr);
}

int STEPattributeList::list_length()
{
    return EntryCount();
}

void STEPattributeList::push(STEPattribute *a)
{
    AttrListNode *saln = new AttrListNode(a);
    AppendNode (saln);
}










/*
STEPattributeListNode NilSTEPattributeListNode;

class init_NilSTEPattributeListNode
{
public:
  inline init_NilSTEPattributeListNode() 
  {
    NilSTEPattributeListNode.tl = &NilSTEPattributeListNode;
    NilSTEPattributeListNode.ref = -1;
  }
};

static init_NilSTEPattributeListNode NilSTEPattributeListNode_initializer;

*/
