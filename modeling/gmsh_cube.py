# This code build and meshes hollow cubes

import gmsh
import math
import os
import sys
import numpy as np

thickness_list = np.linspace(0, 0.013, num=500) # Choose thickness intervals

# Making sure the output thickness file is in a python list format
with open('Computational Homogenization/Meshes/hollow_cube_thickness_n500.txt', 'a') as f:
        f.write("[")
        f.close()

index = 0
# Loop through the thicknesses
for thickness in thickness_list:

    gmsh.initialize()

    t = thickness

    # Build boxes
    gmsh.model.occ.addBox(-0.01, -0.01, -0.01, 0.02, 0.02, 0.02)
    gmsh.model.occ.addBox(-0.009+t/2, -0.009+t/2, -0.009+t/2, 0.018-t, 0.018-t, 0.018-t)
    # Cut out inner box
    gmsh.model.occ.cut([(3, 1)], [(3, 2)])
    gmsh.model.occ.synchronize()

    # Set mesh size
    lc = 1e-3
    gmsh.model.mesh.setSize([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)], lc)

    # Translation matrices for the three dimensions
    translationX = [1, 0, 0, 0.02, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] # Translation in x
    translationY = [1, 0, 0, 0, 0, 1, 0, 0.02, 0, 0, 1, 0, 0, 0, 0, 1] # Translation in y
    translationZ = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.02, 0, 0, 0, 1] # Translation in z

    # Set periodic mesh. Copies the mesh of the opposing side
    gmsh.model.mesh.setPeriodic(2, [6], [1], translationX)

    gmsh.model.mesh.setPeriodic(2, [4], [2], translationY)

    gmsh.model.mesh.setPeriodic(2, [3], [5], translationZ)

    gmsh.model.mesh.generate(3)

    # Save mesh-file
    gmsh.write(f"Computational Homogenization/Meshes/hollow_cubes_n500/hollow_cube_periodic{index}.mesh")
    index += 1

    gmsh.finalize()

    # Write thickness to a txt-file
    with open('Computational Homogenization/Meshes/hollow_cube_thickness_n500.txt', 'a') as f:
        f.write(str(t))
        f.write(",\n")
        f.close()

# Making sure the thickness txt-file has a python syntax
with open('Computational Homogenization/Meshes/hollow_cube_thickness_n500.txt', 'a') as f:
    f.write("]")
    f.close()

print([round(thickness, 5) for thickness in thickness_list])

# Launch the GUI to see the results:
# if '-nopopup' not in sys.argv:
#     gmsh.fltk.run()
