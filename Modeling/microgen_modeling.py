# This code builds TPMS STL models
# Requires Microgen Python library

import microgen
import cadquery as cq
import numpy as np

thickness_list = np.linspace(0.05, 0.797, num=200) # Choose thickness intervals
for i in thickness_list:
    print(f'{np.round(i, 3)}, ')
isovalues = []

# Loop thicknesses
for thick in thickness_list:

    thickness = thick
    isovalues.append(thickness * np.pi) # Microgen isovalue definition

    geometry = microgen.Tpms(
        surface_function=microgen.tpms.schoenIWP, # Change geometry here
        type_part="sheet",
        thickness=thickness
    )

    ### Export to STL
    shape = geometry.generate(nSample=20) # Increase nSample for refined mesh.
    cq.exporters.export(shape, "Modeling/microgen_IWPs/microgen_IWP_n200/microgen_IWP_" + str(thickness) + ".STL")
    ##

# Write isovalues to txt file
with open('Modeling/microgen_IWPs/isovalues_IWP_n200.txt', 'a') as f:
    f.write("[")
    for isovalue in isovalues:
        f.write(str(isovalue))
        f.write(",\n")
    f.write("]")
f.close()
