#-------------------------------------------------------------------------
# DESCRIPTION

# Input file for computing the temperature distribution in a
# tokamak divertor monoblock 
# The monoblock is typically comprised of a copper-chromium-zirconium (CuCrZr)
# pipe surrounded by tungsten armour with an OFHC copper pipe interlayer in
# between. This simplified model is comprised of a solid/filled OFHC copper
# cylinder by surrounded by tungsten armour; the CuCrZr pipe is not included
# and coolant flow is not modelled.
# The mesh uses first order elements with a nominal mesh refinement of one 
# division per millimetre.
# The boundary conditions include:
# - A constant heat flux applied to the top surface (simulating plasma interaction)
# - Convective cooling on the inner pipe surface (simulating coolant flow)
# - Insulated side walls

# The solve is steady-state and outputs the temperature distribution,
# with a focus on the maximum temperature in the structure.

# Material properties are defined as constants based on specific temperatures:
# - CuCrZr properties at 200°C
# - OFHC Copper properties at 250°C
# - Tungsten properties at 600°C
#-------------------------------------------------------------------------
# PARAMETER DEFINITIONS

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# File handling
name=simple_monoblock
outputDir=outputs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Geometry
PI=3.141592653589793

intLayerExtDiam=17e-3 # m
intLayerExtCirc=${fparse PI * intLayerExtDiam}

monoBThick=3e-3     # m
monoBWidth=${fparse intLayerExtDiam + 2*monoBThick}
monoBDepth=12e-3      # m
monoBArmHeight=8e-3   # m

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mesh Sizing
meshRefFact=1
meshDens=1e3 # divisions per metre (nominal)

# Mesh Order
secondOrder=false
orderString=FIRST

# Note: some of the following values must be even integers. This is in some
# cases a requirement for the meshing functions used, else is is to ensure a
# division is present at the centreline, thus allowing zero-displacement
# boundary conditions to be applied to the centre node. These values are
# halved, rounded to int, then doubled to ensure the result is an even int.

# Number of divisions along the top section of the monoblock armour.
monoBArmDivs=${fparse int(monoBArmHeight * meshDens * meshRefFact)}

# Number of divisions around each quadrant of the circumference of the cylinder
# and radial section of the monoblock armour.
pipeCircSectDivs=${fparse 2 * int(monoBWidth/2 * meshDens * meshRefFact / 2)}

# Number of radial divisions for the interlayer and radial section of the
# monoblock armour respectively.
intLayerRadDivs=${
  fparse max(int(intLayerExtDiam/2 * meshDens * meshRefFact), 5)
}
monoBRadDivs=${
  fparse max(int((monoBWidth-intLayerExtDiam)/2 * meshDens * meshRefFact), 5)
}

# Number of divisions along monoblock depth (i.e. z-dimension).
extrudeDivs=${fparse max(2 * int(monoBDepth * meshDens * meshRefFact / 2), 4)}

monoBElemSize=${fparse monoBDepth / extrudeDivs}
tol=${fparse monoBElemSize / 10}
ctol=${fparse intLayerExtCirc / (8 * 4 * pipeCircSectDivs)}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Material Properties
# Mono-Block/Armour = Tungsten
# Interlayer = Oxygen-Free High-Conductivity (OFHC) Copper

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

coolantTemp=100.0      # degC
heatTransCoeff=125.0e3 # W.m^-2.K^-1
surfHeatFlux=10.0e6    # W.m^-2

# Copper-Chromium-Zirconium pg.148 at 200degC
cucrzrDensity = 8816.0  # kg.m^-3
cucrzrThermCond = 343.0 # W.m^-1.K^-1
cucrzrSpecHeat = 407.0  # J.kg^-1.K^-1

# Pure Copper pg.134 at 250degC
cuDensity = 8829.0  # kg.m^-3
cuThermCond = 384.0 # W.m^-1.K^-1
cuSpecHeat = 406.0  # J.kg^-1.K^-1

# Tungsten pg.224 at 600degC
wDensity = 19150.0  # kg.m^-3
wThermCond = 127.0 # W.m^-1.K^-1
wSpecHeat = 147.0  # J.kg^-1.K^-1



#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------



[Mesh]
  second_order = ${secondOrder}

  [mesh_monoblock]
    type = PolygonConcentricCircleMeshGenerator
    num_sides = 4
    polygon_size = ${fparse monoBWidth / 2}
    polygon_size_style = apothem  # i.e. distance from centre to edge
    ring_radii = ${fparse intLayerExtDiam / 2}
    num_sectors_per_side = '
      ${pipeCircSectDivs}
      ${pipeCircSectDivs}
      ${pipeCircSectDivs}
      ${pipeCircSectDivs}
    '
    ring_intervals = ${intLayerRadDivs}
    background_intervals = ${monoBRadDivs}
    preserve_volumes = on
    flat_side_up = true
    ring_block_names = 'pipe-cucrzr interlayer-cu'
    background_block_names = monoblock
    interface_boundary_id_shift = 1000
    external_boundary_name = monoblock_boundary
    generate_side_specific_boundaries = true
  []

  [mesh_armour]
    type = GeneratedMeshGenerator
    dim = 2
    xmin = ${fparse monoBWidth /-2}
    xmax = ${fparse monoBWidth / 2}
    ymin = ${fparse monoBWidth / 2}
    ymax = ${fparse monoBWidth / 2 + monoBArmHeight}
    nx = ${pipeCircSectDivs}
    ny = ${monoBArmDivs}
    boundary_name_prefix = armour
  []

  [define_bc_surfaces]
    type = SideSetsFromNormalsGenerator
    input = extrude
    normals = '0 1 0  0 -1 0'
    new_boundary = 'bc-top-heatflux bc-pipe-heattransf'
  []

  [combine_meshes]
    type = StitchedMeshGenerator
    inputs = 'mesh_monoblock mesh_armour'
    stitch_boundaries_pairs = 'monoblock_boundary armour_bottom'
    clear_stitched_boundary_ids = true
  []

  [merge_block_names]
    type = RenameBlockGenerator
    input = combine_meshes
    old_block = '3 0'
    new_block = 'armour-w armour-w'
  []

  [merge_boundary_names]
    type = RenameBoundaryGenerator
    input = merge_block_names
    old_boundary = 'armour_top
                    armour_left 10002 15002
                    armour_right 10004 15004
                    10003 15003'
    new_boundary = 'top
                    left left left
                    right right right
                    bottom bottom'
  []

  [extrude]
    type = AdvancedExtruderGenerator
    input = merge_boundary_names
    direction = '0 0 1'
    heights = ${monoBDepth}
    num_layers = ${extrudeDivs}
  []

  [pin_x]
    type = BoundingBoxNodeSetGenerator
    input = define_bc_surfaces
    bottom_left = '${fparse -ctol}
                   ${fparse (monoBWidth/-2)-ctol}
                   ${fparse -tol}'
    top_right = '${fparse ctol}
                 ${fparse (monoBWidth/-2)+ctol}
                 ${fparse (monoBDepth)+tol}'
    new_boundary = bottom_x0
  []
  [pin_z]
    type = BoundingBoxNodeSetGenerator
    input = pin_x
    bottom_left = '${fparse (monoBWidth/-2)-ctol}
                   ${fparse (monoBWidth/-2)-ctol}
                   ${fparse (monoBDepth/2)-tol}'
    top_right = '${fparse (monoBWidth/2)+ctol}
                 ${fparse (monoBWidth/-2)+ctol}
                 ${fparse (monoBDepth/2)+tol}'
    new_boundary = bottom_z0
  []
  [define_full_volume_nodeset]
    type = BoundingBoxNodeSetGenerator
    input = pin_z
    bottom_left = '
      ${fparse (monoBWidth/-2)-ctol}
      ${fparse (monoBWidth/-2)-ctol}
      ${fparse -tol}
    '
    top_right = '
      ${fparse (monoBWidth/2)+ctol}
      ${fparse (monoBWidth/2)+monoBArmHeight+ctol}
      ${fparse monoBDepth+tol}
    '
    new_boundary = volume
  []
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = ${orderString}
    initial_condition = ${coolantTemp}
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
[]



[Materials]
  [cucrzr_thermal]
    type = HeatConductionMaterial
    thermal_conductivity = ${cucrzrThermCond}
    specific_heat = ${cucrzrSpecHeat}
    block = 'pipe-cucrzr'
  []

  [copper_thermal]
    type = HeatConductionMaterial
    thermal_conductivity = ${cuThermCond}
    specific_heat = ${cuSpecHeat}
    block = 'interlayer-cu'
  []

  [tungsten_thermal]
    type = HeatConductionMaterial
    thermal_conductivity = ${wThermCond}
    specific_heat = ${wSpecHeat}
    block = 'armour-w'
  []

  [cucrzr_density]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = ${cucrzrDensity}
    block = 'pipe-cucrzr'
  []

  [copper_density]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = ${cuDensity}
    block = 'interlayer-cu'
  []

  [tungsten_density]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = ${wDensity}
    block = 'armour-w'
  []
[]


[BCs]
  [heat_flux_in]
    type = NeumannBC
    variable = temperature
    boundary = 'bc-top-heatflux'
    value = ${surfHeatFlux}
  []
  [convective_cooling]
    type = ConvectiveHeatFluxBC
    variable = temperature
    boundary = 'bc-pipe-heattransf'  
    T_infinity = ${coolantTemp}
    heat_transfer_coefficient = ${heatTransCoeff}
  []
  [insulated_sides]
    type = NeumannBC
    variable = temperature
    boundary = 'left right'
    value = 0
  []
[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre    boomeramg'
[]

[Postprocessors]
  [max_temp]
    type = NodalExtremeValue
    variable = temperature
  []
[]

[Outputs]
  exodus = true
  [write_to_file]
    type = CSV
    show = 'max_temp'
    file_base = '${outputDir}/${name}_out'
  []
[]