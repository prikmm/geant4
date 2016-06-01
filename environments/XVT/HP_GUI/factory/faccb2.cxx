// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: faccb2.cxx,v 2.1 1998/07/12 02:39:13 urbi Exp $
// GEANT4 tag $Name: geant4-00 $
//
/* 
	This is version 4.5 of XVT-Architect.
	This file was automatically generated by XVT-Architect,
	Do not modify its contents.
*/

#include "factory.h"
#include "XVTPwr.h"
#include "AppDef.h"
#include PwrGen_i
#include "facdec.h"
#include "classes.h"
#include "defines.h"
#include "faccb.h"
#define _PA_REF(x) x=x

void _Init_GUI_faccb2() { }

void* C_CTaskDoc1003_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	CTaskDoc* anInstance = NULL;
	return anInstance;
}

void I_CTaskDoc1003_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData, CDataMembers* theDataMembers)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	_PA_REF(theDataMembers);
	CTaskDoc* anInstance = PtrCast(CTaskDoc, theData);
	if (!anInstance) return;

	
	
	
}

CFactoryElement _CTaskDoc1003DEFAULT(&GUIFactory, CTaskDoc1003, 20043, 1, C_CTaskDoc1003_GUI_DEFAULT, I_CTaskDoc1003_GUI_DEFAULT, G4XvtGUI1002, 0, 1, 2, 0);

void* C_LogBar_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	CSubview* anEnclosure = PtrCast( CSubview, theData );
	CToolBar *anInstance = new CToolBar( anEnclosure, CToolBar::AL_TOP );
	return anInstance;
}

void I_LogBar_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData, CDataMembers* theDataMembers)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	_PA_REF(theDataMembers);
	CToolBar* anInstance = PtrCast(CToolBar, theData);
	if (!anInstance) return;

	
	IPAFactoryView(anInstance, "", NULLcCMenuButton1095md, NULLcmd, TRUE, TRUE, TRUE, LEFTSTICKY, FALSE, FALSE);
	
	
	anInstance->IToolBar(2, 5);
	anInstance->InsertSeparator( 0 );
	anInstance->InsertSeparator( 2 );
	anInstance->InsertSeparator( 3 );
	anInstance->InsertSeparator( 8 );
	anInstance->InsertSeparator( 9 );
	anInstance->InsertSeparator( 10 );
	anInstance->InsertSeparator( 12 );
	anInstance->DoSize( anInstance->GetFrame() );
	
}

CFactoryElement _LogBarDEFAULT(&GUIFactory, LogBar, 20123, 3, C_LogBar_GUI_DEFAULT, I_LogBar_GUI_DEFAULT, G4Win, 0, 1, 5, 1);

void* C_CMenuButton1077_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	CSubview* anEnclosure = PtrCast( CSubview, theData );
	CMenuButton *anInstance = new CMenuButton( anEnclosure, CRect((UNITS)105, (UNITS)3, (UNITS)128, (UNITS)26), STOPcmd );
	return anInstance;
}

void I_CMenuButton1077_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData, CDataMembers* theDataMembers)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	_PA_REF(theDataMembers);
	CMenuButton* anInstance = PtrCast(CMenuButton, theData);
	if (!anInstance) return;

	
	IPAFactoryView(anInstance, "", NULLcmd, NULLcmd, TRUE, TRUE, FALSE, TOPSTICKY|LEFTSTICKY, FALSE, FALSE);
	
	
	anInstance->SetTogglable( FALSE );
	CImage anImage(1004);
	CPicture* aPicture = new CPicture( anInstance, CPoint((UNITS)0, (UNITS)0), anImage );
	_PA_REF(aPicture);
	anInstance->SetCommands( NULLcmd, NULLcmd, STOP_INcmd, STOP_OUTcmd );
	anInstance->SizeToFit();
	
	
}

CFactoryElement _CMenuButton1077DEFAULT(&GUIFactory, CMenuButton1077, 20126, 3, C_CMenuButton1077_GUI_DEFAULT, I_CMenuButton1077_GUI_DEFAULT, LogBar, 0, 1, 13, 4);

void* C_commandBox_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	CSubview* anEnclosure = PtrCast( CSubview, theData );
	NLineText* anInstance = new NLineText(anEnclosure, CPoint((UNITS)230, (UNITS)3), (UNITS)425);
	return anInstance;
}

void I_commandBox_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData, CDataMembers* theDataMembers)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	_PA_REF(theDataMembers);
	NLineText* anInstance = PtrCast(NLineText, theData);
	if (!anInstance) return;

	{
		CFont aFont;
		aFont.Deserialize("01\\system\\0\\12\\");
		CEnvironment anEnv(0xaaaa, 0xffffff, 0xbffffff, PAT_SOLID		, 0x7000000, PAT_SOLID,1, aFont, M_COPY, P_SOLID, FALSE);
		
		anInstance->SetEnvironment(anEnv);
	}

	
	IPAFactoryView(anInstance, "", NULLcmd, NULLcmd, TRUE, TRUE, FALSE, TOPSTICKY|LEFTSTICKY, FALSE, FALSE);
	
	unsigned anAttribute = TX_AUTOHSCROLL|TX_AUTOVSCROLL|TX_BORDER|TX_WRAP|0;
	if ((anInstance->GetCWindow()->GetAttributes() & WSF_NO_MENUBAR) != 0)
		anAttribute |= TX_NOMENU;
	anInstance->INativeTextEdit(anAttribute, (UNITS)1000 ,1000, STRING5, FALSE,anInstance->IsVisible(), anInstance->GetGlue());
	anInstance->SetTabSize(4);
	
	
}

CFactoryElement _commandBoxDEFAULT(&GUIFactory, commandBox, 20040, 3, C_commandBox_GUI_DEFAULT, I_commandBox_GUI_DEFAULT, LogBar, 0, 1, 13, 9);

void* C_CommandLog_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	CSubview* anEnclosure = PtrCast( CSubview, theData );
	NScrollText* anInstance = new NScrollText(anEnclosure, CRect((UNITS)485, (UNITS)30, (UNITS)966, (UNITS)255), TRUE, TRUE);

	return anInstance;
}

void I_CommandLog_GUI_DEFAULT(const PAFactory* theFactory, CObjectRWC* theData, CDataMembers* theDataMembers)
{
	_PA_REF(theFactory);
	_PA_REF(theData);
	_PA_REF(theDataMembers);
	NScrollText* anInstance = PtrCast(NScrollText, theData);
	if (!anInstance) return;

	{
		CFont aFont;
		aFont.Deserialize("01\\system\\0\\12\\");
		CEnvironment anEnv(0xffffff, 0x7000000, 0xffffff, PAT_SOLID		, 0x7000000, PAT_SOLID,1, aFont, M_COPY, P_SOLID, FALSE);
		
		anInstance->SetEnvironment(anEnv);
	}

	
	IPAFactoryView(anInstance, "", NULLcmd, NULLcmd, TRUE, TRUE, FALSE, TOPSTICKY|RIGHTSTICKY|LEFTSTICKY|BOTTOMSTICKY, FALSE, FALSE);
	
	unsigned anAttribute = TX_BORDER|TX_WRAP|0;
	if ((anInstance->GetCWindow()->GetAttributes() & WSF_NO_MENUBAR) != 0)
		anAttribute |= TX_NOMENU;
	anInstance->INativeTextEdit(anAttribute, (UNITS)1000 ,1000, STRING4, FALSE,anInstance->IsVisible(), anInstance->GetGlue());
	anInstance->SetTabSize(4);
	
	
	anInstance->SetVIncrements((UNITS)1 , (UNITS)10);
	anInstance->SetHIncrements((UNITS)10 , (UNITS)50);
	
}

CFactoryElement _CommandLogDEFAULT(&GUIFactory, CommandLog, 20041, 3, C_CommandLog_GUI_DEFAULT, I_CommandLog_GUI_DEFAULT, G4Win, 0, 1, 5, 3);

