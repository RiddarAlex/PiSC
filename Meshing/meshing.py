# This file meshes an STL file

import gmsh
import math
import os
import sys
import numpy as np
import pymeshlab
ms = pymeshlab.MeshSet()

def createGeometryAndMesh(filename, count):
    # Clear all models and merge an STL mesh that we would like to remesh (from
    # the parent directory):
    gmsh.clear()
    path = os.path.dirname(os.path.abspath(__file__))
    IWP = 'Modeling/microgen_IWPs/microgen_IWP_n200/' + filename + '.stl'
    gmsh.merge(os.path.join(path, IWP))

    # We first classify ("color") the surfaces by splitting the original surface
    # along sharp geometrical features. This will create new discrete surfaces,
    # curves and points.

    # Angle between two triangles above which an edge is considered as sharp,
    # retrieved from the ONELAB database (see below):
    angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]

    # For complex geometries, patches can be too complex, too elongated or too
    # large to be parametrized; setting the following option will force the
    # creation of patches that are amenable to reparametrization:
    forceParametrizablePatches = gmsh.onelab.getNumber(
        'Parameters/Create surfaces guaranteed to be parametrizable')[0]

    # For open surfaces include the boundary edges in the classification
    # process:
    includeBoundary = True

    # Force curves to be split on given angle:
    curveAngle = 180
    curveAngleParam = curveAngle * math.pi / 180.
    angleParam = angle * math.pi / 180.
    gmsh.model.mesh.classifySurfaces(angle*math.pi/180, includeBoundary,
                                     forceParametrizablePatches,
                                     curveAngle * math.pi / 180.)

    # Create a geometry for all the discrete curves and surfaces in the mesh, by
    # computing a parametrization for each one
    gmsh.model.mesh.createGeometry()

    # Create a volume from all the surfaces
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])

    gmsh.model.geo.synchronize()

    # We specify element sizes imposed by a size field, just because we can :-)
    f = gmsh.model.mesh.field.add("MathEval")
    if gmsh.onelab.getNumber('Parameters/Apply funny mesh size field?')[0]:
        gmsh.model.mesh.field.setString(f, "F", "2*Sin((x+y)/5) + 3")
    else:
        gmsh.model.mesh.field.setString(f, "F", "4")
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    # We now identify corresponding surfaces on the sides of the
    # geometry automatically.

    eps = 1e-3
    # First we get all labels of the surfaces in the x dimension:
    # The parameters are dependent on the size of the model
    sxmin = gmsh.model.getEntitiesInBoundingBox(-0.5 - eps, -0.5 - eps, -0.5 - eps, -0.5 + eps, 0.5 + eps, 0.5 + eps, 2)
    sxmax = gmsh.model.getEntitiesInBoundingBox(0.5 - eps, -0.5 - eps, -0.5 - eps, 0.5 + eps, 0.5 + eps, 0.5 + eps, 2)

    # y dimension
    symin = gmsh.model.getEntitiesInBoundingBox(-0.5 - eps, -0.5 - eps, -0.5 - eps, 0.5 + eps, -0.5 + eps, 0.5 + eps, 2)
    symax = gmsh.model.getEntitiesInBoundingBox(-0.5 - eps, 0.5 - eps, -0.5 - eps, 0.5 + eps, 0.5 + eps, 0.5 + eps, 2)

    # z dimension
    szmin = gmsh.model.getEntitiesInBoundingBox(-0.5 - eps, -0.5 - eps, -0.5 - eps, 0.5 + eps, 0.5 + eps, -0.5 + eps, 2)
    szmax = gmsh.model.getEntitiesInBoundingBox(-0.5 - eps, -0.5 - eps, 0.5 - eps, 0.5 + eps, 0.5 + eps, 0.5 + eps, 2)

    # Write the surface labels to a txt file to be used with replaceLabel.py
    with open("Computational Homogenization/Meshes/microgen_IWP_n200_periodic_tags.txt", 'a') as f:
      f.write(f"[[{szmin[0][1]}, {szmin[1][1]}, {szmin[2][1]}, {szmin[3][1]}], [{szmax[0][1]}, {szmax[1][1]}, {szmax[2][1]}, {szmax[3][1]}], [{symin[0][1]}, {symin[1][1]}, {symin[2][1]}, {symin[3][1]}], [{symax[0][1]}, {symax[1][1]}, {symax[2][1]}, {symax[3][1]}],[{sxmin[0][1]}, {sxmin[1][1]}, {sxmin[2][1]}, {sxmin[3][1]}], [{sxmax[0][1]}, {sxmax[1][1]}, {sxmax[2][1]}, {sxmax[3][1]}]],\n")

    # Translation matrices for the different dimensions
    translationX = [1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] # Translation in x
    translationY = [1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1] # Translation in y
    translationZ = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 1] # Translation in z

    for i in sxmin:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
        print(f"\nXMIN: {xmin}\nXMAX: {xmax}")
    #     # We translate the bounding box to the right and look for surfaces inside
    #     # it:
        sxmax = gmsh.model.getEntitiesInBoundingBox(xmin - eps + 0, ymin - eps,
                                                    zmin - eps, xmax + eps + 0,
                                                    ymax + eps, zmax + eps, 2)

        print(f"\nSXMIN: {sxmin}\nSXMAX: {sxmax}\n")
    #     # For all the matches, we compare the corresponding bounding boxes...
        for j in sxmax:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j[0], j[1])
            print(f"\nXMIN2: {xmin2}\nXMAX2: {xmax2}")
            xmin2 -= 1
            xmax2 -= 1
            print(f"\nXMIN2: {xmin2}\nXMAX2: {xmax2}")
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translationX)

    for i in symin:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
    #     # We translate the bounding box to the right and look for surfaces inside
    #     # it:
        symax = gmsh.model.getEntitiesInBoundingBox(xmin - eps + 0, ymin - eps,
                                                    zmin - eps, xmax + eps + 0,
                                                    ymax + eps, zmax + eps, 2)

    #     # For all the matches, we compare the corresponding bounding boxes...
        for j in symax:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j[0], j[1])
            xmin2 -= 1
            xmax2 -= 1
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translationY)

    for i in szmin:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
        print(f"\nXMIN: {xmin}\nXMAX: {xmax}")
    #     # We translate the bounding box to the right and look for surfaces inside
    #     # it:
        szmax = gmsh.model.getEntitiesInBoundingBox(xmin - eps + 0, ymin - eps,
                                                    zmin - eps, xmax + eps + 0,
                                                    ymax + eps, zmax + eps, 2)

    #     # For all the matches, we compare the corresponding bounding boxes...
        for j in szmax:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j[0], j[1])
            print(f"\nXMIN2: {xmin2}\nXMAX2: {xmax2}")
            xmin2 -= 1
            xmax2 -= 1
            print(f"\nXMIN2: {xmin2}\nXMAX2: {xmax2}")
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translationZ)

    # Set mesh size
    # gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.06)
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.000000001) 

    # Save mesh-file
    newName = 'microgen_IWP_' + f"{count}"
    gmsh.model.mesh.generate(3)
    gmsh.write('Computational Homogenization/Meshes/microgen_IWPs_n200_periodic/' + newName + '_periodic.mesh')

count_list = [i for i in range(200)]
thickness_list = np.linspace(0.05, 0.797, num=200) # List of thicknesses

for i in count_list:
  thickness = thickness_list[i]
  count = i
  print(thickness)

  gmsh.initialize()

  # Create ONELAB parameters with remeshing options:
  gmsh.onelab.set("""[
    {
      "type":"number",
      "name":"Parameters/Angle for surface detection",
      "values":[40],
      "min":20,
      "max":120,
      "step":1
    },
    {
      "type":"number",
      "name":"Parameters/Create surfaces guaranteed to be parametrizable",
      "values":[0],
      "choices":[0, 1]
    },
    {
      "type":"number",
      "name":"Parameters/Apply funny mesh size field?",
      "values":[0],
      "choices":[0, 1]
    }
  ]""")

  # Name of the 3D-model files
  filename = 'microgen_IWP_' + f"{thickness}"

  # Create the geometry and mesh it:
  createGeometryAndMesh(filename, count)

  # Launch the GUI and handle the "check" event to recreate the geometry and mesh
  # with new parameters if necessary:
  # def checkForEvent():
  #     action = gmsh.onelab.getString("ONELAB/Action")
  #     if len(action) and action[0] == "check":
  #         gmsh.onelab.setString("ONELAB/Action", [""])
  #         createGeometryAndMesh()
  #         gmsh.graphics.draw()
  #     return True

  # if "-nopopup" not in sys.argv:
  #     gmsh.fltk.initialize()
  #     while gmsh.fltk.isAvailable() and checkForEvent():
  #         gmsh.fltk.wait()

  gmsh.finalize()
