import numpy as np
from material_properties import get_steel_properties, get_sodium_properties, get_wick_properties, get_vc_properties
from matplotlib import pyplot as plt

T_plot = np.linspace(280, 1000, 1000)
Na = get_vc_properties()
plt.plot(T, Na['thermal_conductivity'](T))
plt.xlabel('Temperature (K)')
plt.ylabel('Property')
plt.grid(True)
plt.show()