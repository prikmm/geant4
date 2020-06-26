//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
//---------------------------------------------------------------------------
//
// GEANT4 Class file
//
// Description: Data on stopping power
//
// Description: Data on stopping power
//
// Author:        Alexander Bagulya & Vladimir Ivanchenko
//
// Creation date: 23.04.2018
// 
//----------------------------------------------------------------------------
//

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "G4LindhardSorensenData.hh" 
#include "G4PhysicsVector.hh"
#include "G4PhysicsLinearVector.hh"
#include "G4Log.hh"
#include "G4Pow.hh"

const G4int zlist[9] = {1, 10, 18, 36, 54, 66, 79, 92, 109};
const G4int LVECT = 8;
const G4int NPOINT = 41;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4LindhardSorensenData::G4LindhardSorensenData()
{
  g4calc = G4Pow::GetInstance();
  InitialiseData();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4LindhardSorensenData::~G4LindhardSorensenData()
{
  for(G4int i=0; i<=LVECT; ++i) { delete data[i]; }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double G4LindhardSorensenData::GetDeltaL(G4int Z, G4double gamma) const
{
  G4int idx = 0;
  for(; idx<LVECT; ++idx) {
    if(Z < zlist[idx+1]) { break; }
  }
  idx = std::min(idx, LVECT);

  G4double x = G4Log(gamma - 1.0);
  G4double y = ComputeDeltaL(idx, x);

  // interpolation over Z if needed
  if(idx < LVECT && Z > zlist[idx]) {
    G4double y1 = ComputeDeltaL(idx+1, x);
    //G4cout << "idx= " << idx << " x= " << x << " y= " << y << " y1= " << y1 << G4endl;
    y += (y1 - y)*(Z - zlist[idx])/(G4double)(zlist[idx+1] - zlist[idx]);
  }
  y *= g4calc->Z23(Z);
  return y;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double G4LindhardSorensenData::ComputeDeltaL(G4int idx, G4double x) const
{
  G4double y(0.0);
  if(x < xmin) {
    G4double x1 = (data[idx])->Energy(1);
    G4double ymin = (*(data[idx]))[0];
    G4double y1 = (*(data[idx]))[1];
    y = ymin + (y1 - ymin)*(x - xmin)/(x1 - xmin);
  } else if(x >= xmax) {
    G4double x1 = (data[idx])->Energy(NPOINT-2);
    G4double ymax = (*(data[idx]))[NPOINT-1];
    G4double y1 = (*(data[idx]))[NPOINT-2];
    y = y1 + (ymax - y1)*(x - x1)/(xmax - x1);
  } else {
    y = (data[idx])->Value(x); 
  }
  return y;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
 
void G4LindhardSorensenData::InitialiseData()
{
  xmin = G4Log(0.02);
  xmax = G4Log(316.22777);

  const G4double lsdata[9][NPOINT] = {
    {0.0036181621,
     0.0042618872, 0.0040786701, 0.0039097273, 0.0041240731, 0.0044311195,   // 0-5
     0.0059858073, 0.0079655897, 0.0089669217,  0.010091248, 0.0096319233,   // 5-10
     0.0085344428, 0.0087326058,  0.009780094,  0.010617094,  0.011277997,   // 10-15
     0.01185287,  0.012242278,  0.012507042,  0.012369698,  0.012202436,     // 15-20
     0.012016446,  0.011793182,  0.011612196,  0.011375017,  0.010624408,    // 20-25
     0.0096685612, 0.0093765113, 0.0091152632, 0.0087458522, 0.0082645153,   // 25-30
     0.0068625676, 0.0046558921, 0.0021660968, -0.00072760644, -0.0062653709,// 30-35
     -0.015834368,  -0.03224412, -0.062296044,  -0.10327705,  -0.16545368},
    {-0.024837796,
     -0.018943357, -0.013251703, -0.007646936, -0.003751624, 7.4932758e-05,  // 0-5
     0.0027996278, 0.0057534909, 0.0078441157,  0.010341958,  0.012419648,   // 5-10
     0.01491866,  0.016403769,  0.018106955,  0.019249728,  0.020704966,     // 10-15
     0.021636244,  0.022526456,  0.023429501,  0.024251631,  0.024727856,    // 15-20
     0.024997337,  0.025225657,  0.025335138,  0.025474553,  0.025306356,    // 20-25
     0.025048151,   0.02467553,   0.02416496,  0.023519917,  0.022699646,    // 25-30
     0.020902526,  0.017645158,  0.013219058, 0.0059989118, -0.0042559395,   // 30-35
     -0.020223315, -0.044480728, -0.074942757,   -0.1108863,  -0.15439805},
    {-0.04948514,
     -0.039259418, -0.029641846, -0.022112859,   -0.0152972, -0.0077213168,  // 0-5
     -0.0031032447, 0.0023506153, 0.0058573025, 0.0097554723,  0.012692892,  // 5-10
     0.016404575,  0.018642388,  0.021492082,  0.023328047,  0.025358859,    // 10-15
     0.026839018,  0.028169751,  0.029240668,  0.030035325,  0.030689491,    // 15-20
     0.031113343,  0.031518392,  0.031657748,  0.031794607,  0.031672646,    // 20-25
     0.031507453,  0.031024952,  0.030410522,  0.029266343,  0.027729316,    // 25-30
     0.025132583,   0.02141173,  0.016127626,  0.008039769, -0.0046793613,   // 30-35
     -0.02196456, -0.043877567, -0.070852217,  -0.09979099,  -0.13187648},
    {-0.07502957,
     -0.063927817, -0.053895181, -0.043621797, -0.033329203, -0.024765223,   // 0-5
     -0.015996122, -0.0084995741, -0.00017499271, 0.0064195209,   0.01117532,// 5-10
     0.016956954,   0.02155737,  0.025213602,  0.029869549,  0.031980316,    // 10-15
     0.034664781,  0.036995048,  0.038244843,  0.039836367,  0.040930356,    // 15-20
     0.041677336,  0.042211864,  0.042472572,  0.042617755,  0.042393662,    // 20-25
     0.042049893,  0.041341375,  0.040408112,   0.03867031,  0.036457345,    // 25-30
     0.032742289,  0.027897265,  0.020305954, 0.0099602019, -0.0036658833,   // 30-35
     -0.020344557, -0.039700638, -0.059781744, -0.080649004,  -0.10263347},
    {-0.081885964,
     -0.072049323,  -0.06299877, -0.053400445, -0.043830227,  -0.03415406,   // 0-5
     -0.024723298, -0.015706543, -0.0070305988, 0.0016621818, 0.0093560784,  // 5-10
     0.017338568,  0.023286722,  0.028930541,  0.034048359,  0.037750635,    // 10-15
     0.041512165,   0.04362858,  0.046323682,  0.048044285,  0.049379817,    // 15-20
     0.050501289,  0.050963981,  0.051463496,  0.051292532,  0.051074821,    // 20-25
     0.050350498,  0.049370817,  0.047660517,  0.045247985,   0.04188952,    // 25-30
     0.037009647,  0.030171627,  0.021186013, 0.0092064517, -0.0051358689,   // 30-35
     -0.01977885, -0.036354277, -0.053187271, -0.069851489, -0.087776477},
    {-0.083068958,
     -0.074302116, -0.065040071, -0.056500699, -0.047189921,  -0.03781489,   // 0-5
     -0.028666422,   -0.0192974, -0.0097949279, -0.0011649946, 0.0071301285, // 5-10
     0.01548354,  0.023483416,  0.030515545,  0.036162102,  0.040602642,     // 10-15
     0.045011738,  0.048259036,  0.051136495,  0.053524791,  0.054971064,    // 15-20
     0.056247898,  0.056729496,  0.057135059,  0.056923775,   0.05657171,    // 20-25
     0.055504504,  0.054134086,  0.051812977,  0.048857213,  0.044525663,    // 25-30
     0.038593442,   0.03027527,  0.020452704, 0.0082978722, -0.0045556908,   // 30-35
     -0.018750622, -0.033642805, -0.048788529, -0.064509221, -0.080215679},
    {-0.083061344,
     -0.075180377, -0.066630961, -0.058114277, -0.049303999, -0.040618218,   // 0-5
     -0.031608369, -0.022553882, -0.012699373, -0.0033434259, 0.0051129584,  // 5-10
     0.013817276,  0.023132109,  0.030886148,  0.037688311,  0.043478106,    // 10-15
     0.048602814,  0.052502168,  0.055969817,  0.058142903,  0.060860174,    // 15-20
     0.061732795,  0.062844014,  0.062911637,   0.06278651,  0.061898673,    // 20-25
     0.060500452,  0.058323088,  0.055297604,  0.051242701,   0.04596475,    // 25-30
     0.038863441,  0.030302781,  0.019055839, 0.0072837125, -0.0047516889,   // 30-35
     -0.017733076, -0.031547911,  -0.04523347, -0.059727542, -0.074202114},
    {-0.082449782,
     -0.07492693, -0.067041495, -0.059614338, -0.051061455, -0.042587329,    // 0-5
     -0.034242658, -0.025296495, -0.016157818, -0.0064198954, 0.0033148315,  // 5-10
     0.012927383,  0.021914897,   0.03053429,   0.03854699,  0.045293871,    // 10-15
     0.05141546,  0.056473799,  0.060050334,  0.063292547,  0.065215264,     // 15-20
     0.066961692,  0.067369593,  0.067623539,  0.066825274,  0.065740573,    // 20-25
     0.063673411,  0.061033708,  0.057399807,  0.052544441,   0.04583072,    // 25-30
     0.037931379,  0.028569029,  0.017544895, 0.0068751554, -0.0038105519,   // 30-35
     -0.015233268, -0.026956839, -0.038975048, -0.051387898, -0.064344384},
    {-0.081232852,
     -0.074976912, -0.068421001, -0.060951614, -0.053624488, -0.045755024,   // 0-5
     -0.037708151, -0.029246587,  -0.02018635,  -0.01073862, -0.00085387172, // 5-10
     0.0090831897,  0.018970381,   0.02761289,  0.036215831,  0.044355404,   // 10-15
     0.051067631,  0.057143777,  0.061348214,  0.065462308,   0.06715594,    // 15-20
     0.069155687,  0.069845259,  0.069828188,  0.069170195,  0.067585183,    // 20-25
     0.065486015,  0.062437523,  0.058111347,  0.052410482,  0.045597673,    // 25-30
     0.037571853,  0.028019555,  0.016564385, 0.0060254421, -0.0043254115,   // 30-35
     -0.015675051, -0.027353574,  -0.03931666, -0.050325729,  -0.06174447}};

  for(G4int i=0; i<=LVECT; ++i) { 
    data[i] = new G4PhysicsLinearVector(xmin, xmax, NPOINT-1);
    data[i]->SetSpline(true);
    for(std::size_t j=0; j<NPOINT; ++j) {
      data[i]->PutValue(j, lsdata[i][j]);
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
