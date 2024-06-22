# this is a comment

#_* Variables Block
specific_heat = 365.0
thermal_conductivity = 13.5

# Testing comments in the variables block
#**

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 10
  ny = 10
  xmax = 1.0
  ymax = 1.0
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = FIRST
    initial_condition = 139.2462615966797
  []
[]

[Kernels]
  [heat-conduction]
    type = ADHeatConduction
    variable = temperature
  []
  [heat-conduction-dt]
    type = ADHeatConductionTimeDerivative
    variable = temperature
  []
[]

[BCs]
  [convective]
    type = ADConvectiveHeatFluxBC
    variable = temperature
    boundary = 'right'
    T_infinity = '293.15'
    heat_transfer_coefficient = 7.8
  []
  [fixed-temp]
    type = ADDirichletBC
    variable = temperature
    value = 373.15
    boundary = 'left'
  []
[]

[Materials]
  [steel-density]
    type = ADGenericConstantMaterial
    prop_names = 'density'
    prop_values = '7800'
  []
  [steel-conduction]
    type = ADHeatConductionMaterial
    specific_heat = ${specific_heat}
    thermal_conductivity = ${thermal_conductivity}
  []
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  petsc_options = '-snes_ksp_ew'
  petsc_options_iname = '-pc_type -sub_pc_type -pc_asm_overlap -ksp_gmres_restart'
  petsc_options_value = 'asm lu 1 101'
  line_search = 'none'
  nl_abs_tol = 1e-9
  nl_rel_tol = 1e-8
  l_tol = 1e-6
  start_time = 0.0
  dt = 1.0
  num_steps = 40
[]

[Outputs]
  exodus = true
[]