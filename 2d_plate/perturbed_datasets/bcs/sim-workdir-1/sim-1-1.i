#_* Variables Block
end_time = 30
time_step = 0.5
max_temp = 500
init_temp = 50
xmax = 10
ymax = 2
thermal_conductivity = 45
specific_heat = 0.5
prop_values = 8000
bc1_value = ${init_temp}
bc1_boundary = left
bc2_value = ${init_temp}+${max_temp}*(1-exp(-t))
bc2_boundary = right
#**

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = 2
        nx = 20
        ny = 10
        xmax = ${xmax}
        ymax = ${ymax}
    []
[]

[Variables]
    [temperature]
        initial_condition = ${init_temp}
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = temperature
    []
    [time_derivative]
        type = HeatConductionTimeDerivative
        variable = temperature
    []
    [heat_source]
        type = HeatSource
        variable = temperature
        value = 1e4
    []
[]

[Materials]
    [thermal]
        type = HeatConductionMaterial
        thermal_conductivity = ${thermal_conductivity}
        specific_heat = ${specific_heat}
    []
    [density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = ${prop_values}
    []
[]

[BCs]
    [t_left]
        type = DirichletBC
        variable = temperature
        value = ${bc1_value}
        boundary = ${bc1_boundary}
    []
    [t_right]
        type = FunctionDirichletBC
        variable = temperature
        function = ${bc2_value}
        boundary = ${bc2_boundary}
    []
[]

[Executioner]
    type = Transient
    end_time = ${end_time}
    dt = ${time_step}
[]

[Outputs]
    exodus = true
[]
