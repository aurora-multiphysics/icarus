#include "IcarusApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
IcarusApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

IcarusApp::IcarusApp(InputParameters parameters) : MooseApp(parameters)
{
  IcarusApp::registerAll(_factory, _action_factory, _syntax);
}

IcarusApp::~IcarusApp() {}

void
IcarusApp::registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  ModulesApp::registerAllObjects<IcarusApp>(f, af, s);
  Registry::registerObjectsTo(f, {"IcarusApp"});
  Registry::registerActionsTo(af, {"IcarusApp"});

  /* register custom execute flags, action syntax, etc. here */
}

void
IcarusApp::registerApps()
{
  registerApp(IcarusApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
IcarusApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  IcarusApp::registerAll(f, af, s);
}
extern "C" void
IcarusApp__registerApps()
{
  IcarusApp::registerApps();
}
