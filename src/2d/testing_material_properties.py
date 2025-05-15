import numpy as np
from material_properties import get_steel_properties, get_sodium_properties, get_wick_properties, get_vc_properties
from matplotlib import pyplot as plt
from params import get_all_params
from mesh import generate_composite_mesh

Na_props = get_sodium_properties()
steel_props = get_steel_properties()
params = get_all_params()
mesh = generate_composite_mesh(params, params)

T_plot = np.linspace(280, 1000, 1000)
property = get_wick_properties()
plt.plot(T_plot, property['specific_heat'](T_plot, Na_props, steel_props, params))
plt.xlabel('Temperature (K)')
plt.ylabel('Property')
plt.grid(True)
plt.show()