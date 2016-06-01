#include "G4BSplineCurveWithKnotsCreator.hh"
#include "G4BSplineCurve.hh"
typedef RWTValVector<G4double> G4doubleVector;

G4BSplineCurveWithKnotsCreator G4BSplineCurveWithKnotsCreator::csc;


G4BSplineCurveWithKnotsCreator::G4BSplineCurveWithKnotsCreator()
{
  G4GeometryTable::RegisterObject(this);
}


G4BSplineCurveWithKnotsCreator::~G4BSplineCurveWithKnotsCreator(){}


void G4BSplineCurveWithKnotsCreator::CreateG4Geometry(STEPentity& Ent)
{
  // Created by L. Broglia

  G4int             degree;
  G4Point3DVector   controlPointsList;
  G4doubleVector    knots;
  G4doubleVector    weightsData;
  G4BSplineCurve*   bSpline = new G4BSplineCurve();

  // degree
  G4String attrName("degree");
  STEPattribute *Attr = GetNamedAttribute(attrName, Ent);
  degree = *Attr->ptr.i;
  

  // controlPointsList
  attrName = "control_points_list";
  Attr = GetNamedAttribute(attrName, Ent);
   
  // Loop to find the entities
  // Fill points
  // Temporary solution until the STEP toolkit has been updated:
  char c = ' ';
  STEPaggregate *Aggr = Attr->ptr.a;
  char tmp[16];
  SCLstring s;
  const char *Str = Aggr->asStr(s);
  G4int Count=0;
  G4int nbpoint = 0;
  STEPentity *Entity;
  G4int stringlength = strlen(Str);  

  while(c != ')')
  {  
    while(c != '#')
    {
      c = Str[Count];
      Count++;
      
      if(Count>stringlength)
      {
	G4cout << "\nString index overflow in G4ControlPoints:116";
	exit(0);
      }
    }
    
    c = Str[Count];
    int Index=0;
    
    while(c != ',' && c != ')')
    {
      tmp[Index]=c;
      Index++;
      Count++;
      c = Str[Count];
    }

    tmp[Index]='\0'; 
    //c = ' ';
    Index = atoi(tmp);
  
    MgrNode* MgrTmp = instanceManager.FindFileId(Index);
    Index = instanceManager.GetIndex(MgrTmp);
    Entity = instanceManager.GetSTEPentity(Index);

    void *tmp =G4GeometryTable::CreateObject(*Entity);
    G4Point3D* pt = (G4Point3D*) tmp;
    G4Point3D Pt(pt->x(), pt->y(), pt->z());

    if(controlPointsList.length() <= nbpoint+1)
      controlPointsList.reshape(nbpoint+1);

    controlPointsList[nbpoint] = Pt;
    
    nbpoint++;
  }
  
  
  // weightsData
  attrName = "knot_multiplicities";
  Attr = GetNamedAttribute(attrName, Ent);
  c = ' ';
  Aggr = Attr->ptr.a;
  Str = Aggr->asStr(s);
  nbpoint = 0;
  Count = 0;

  while(c != ')')
  {      
    Count++;
    while(c == '(')
    {
      c = Str[Count];
      Count++;
    }

    int Index=0;
    c = Str[Count];

    while( c != ',' && c != ')' )
    {
      tmp[Index]=c;
      Index++;
      Count++;
      c = Str[Count];
    }

    tmp[Index]='\0';

    // L. Broglia : I am not sure of the function "atoi"
    G4double weight = atoi(tmp);
    if (weight <= 0)
      weight = 1;

    if(weightsData.length() <= nbpoint+1)
      weightsData.reshape(nbpoint+1);
    
    weightsData[nbpoint] = weight;
    nbpoint++;
  }


  // knots
  attrName = "knots";
  Attr = GetNamedAttribute(attrName, Ent); 
  c = ' ';
  Aggr = Attr->ptr.a;
  Str = Aggr->asStr(s);
  nbpoint = 0;
  Count = 0;

  while(c != ')')
  {  
    Count++;
    
    while(c == '(')
    {
      c = Str[Count];
      Count++;
    }

    int Index=0;
    c = Str[Count];

    while(c != ',' && c != ')')
    {
      tmp[Index]=c;
      Index++;
      Count++;
      c = Str[Count];
    }

    tmp[Index]='\0';
   
    // L. Broglia : I am not sure of the function "atoi"
    G4double knot = atoi(tmp);
    if (knot <= 0)
      knot = 1;

    if(knots.length() <= nbpoint+1)
      knots.reshape(nbpoint+1);
   
    knots[nbpoint] = knot;
    nbpoint++;
  }

  bSpline->Init(degree, &controlPointsList, &knots, &weightsData);
  
  createdObject = bSpline;

}

void G4BSplineCurveWithKnotsCreator::CreateSTEPGeometry(void* G4obj)
{

}
