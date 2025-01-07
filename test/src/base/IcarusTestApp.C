//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "IcarusTestApp.h"
#include "IcarusApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
IcarusTestApp::validParams()
{
  InputParameters params = IcarusApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

IcarusTestApp::IcarusTestApp(InputParameters parameters) : MooseApp(parameters)
{
  IcarusTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

IcarusTestApp::~IcarusTestApp() {}

void
IcarusTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  IcarusApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"IcarusTestApp"});
    Registry::registerActionsTo(af, {"IcarusTestApp"});
  }
}

void
IcarusTestApp::registerApps()
{
  registerApp(IcarusApp);
  registerApp(IcarusTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
IcarusTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  IcarusTestApp::registerAll(f, af, s);
}
extern "C" void
IcarusTestApp__registerApps()
{
  IcarusTestApp::registerApps();
}
