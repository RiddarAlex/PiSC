""" INCOMPLETE Code for generating TPMS in Gmsh using its CAD kernel using the 
marching cubes algorithm """

# Marching cubes for TPMS in Gmsh
# marchingGmsh.py

import gmsh
import math
import sys
import numpy as np

pi = math.pi
sin = lambda x : math.sin(x)
cos = lambda x : math.cos(x)


gmsh.initialize()

def createGeometryAndMesh():
    # Clear all models and create a new one
    gmsh.clear()
    gmsh.model.add("model_one")

    x_res = 15   # Resolution of the 3D grid. Number of points along x, y, z of the unit cell
    y_res = 15   # Keep these below 100, must be below 3 digits!
    z_res = 15
    xSpace = np.linspace(-0.5, 0.5, x_res)
    ySpace = np.linspace(-0.5, 0.5, y_res)
    zSpace = np.linspace(-0.5, 0.5, z_res)

    F = lambda x, y, z : cos(2*pi*x)*sin(2*pi*y) + cos(2*pi*y)*sin(2*pi*z) + cos(2*pi*z)*sin(2*pi*x)
    C = lambda x, y, z : 0.6

    """
    # Grid points
    for i in range(x_res+1):
        for j in range(y_res+1):
            for k in range(z_res+1):
                # Create the tag for each point
                tag_list = [str(i), str(j), str(k)]
                for _ in range(3):
                    if len(tag_list[_]) == 1:
                        tag_list[_] = f'0{tag_list[_]}'
                tag_string = f'9{tag_list[0]}{tag_list[1]}{tag_list[2]}'
                point_tag = int(tag_string)
                
                # Create grid points with tags defined above
                gmsh.model.geo.addPoint(0+i/x_res, 0+j/y_res, 0+k/z_res, tag=point_tag)
    gmsh.model.geo.synchronize()
    """


    # Cube march
    C = C(0, 0, 0)
    for i in range(-x_res, x_res):
        for j in range(-y_res, y_res):
            for k in range(-z_res, z_res):
                # Function values on every corner of the marching cube
                D_value = F(2*(0+i)/x_res, 2*(0+j)/y_res, 2*(0+k)/z_res)
                C_value = F(2*(1+i)/x_res, 2*(0+j)/y_res, 2*(0+k)/z_res)
                A_value = F(2*(0+i)/x_res, 2*(1+j)/y_res, 2*(0+k)/z_res) 
                B_value = F(2*(1+i)/x_res, 2*(1+j)/y_res, 2*(0+k)/z_res) 
                H_value = F(2*(0+i)/x_res, 2*(0+j)/y_res, 2*(1+k)/z_res)
                G_value = F(2*(1+i)/x_res, 2*(0+j)/y_res, 2*(1+k)/z_res)
                E_value = F(2*(0+i)/x_res, 2*(1+j)/y_res, 2*(1+k)/z_res) 
                F_value = F(2*(1+i)/x_res, 2*(1+j)/y_res, 2*(1+k)/z_res) 
                

                edge0coords  = 
                edge1coords  = 
                edge2coords  = 
                edge3coords  = 
                edge4coords  = 
                edge5coords  = 
                edge6coords  = 
                edge7coords  = 
                edge8coords  = 
                edge9coords  = 
                edge10coords = 
                edge11coords = 

                corner_activation_string = ''
                for q in [H_value, G_value, F_value, E_value, D_value, C_value, B_value, A_value]:
                    if -C <= q and q <= C:
                        corner_activation_string += '1'
                    else:
                        corner_activation_string += '0'
                
                triang_index = int(corner_activation_string, 2)
                triangulation = getTriangulation(triang_index)



                for element in triangulation:
                # Skapa punkter och kanter och ytor
                    if triangulation[0] != -1:
                        pass
                
            

            
createGeometryAndMesh()
            
                    






# Launch the GUI and handle the "check" event (recorded in the "ONELAB/Action"
# parameter) to recreate the geometry with a new twisting angle if necessary:
def checkForEvent():
    action = gmsh.onelab.getString("ONELAB/Action")
    if len(action) and action[0] == "check":
        gmsh.onelab.setString("ONELAB/Action", [""])
        createGeometryAndMesh()
        gmsh.graphics.draw()
    return True

if "-nopopup" not in sys.argv:
    gmsh.fltk.initialize()
    while gmsh.fltk.isAvailable() and checkForEvent():
        gmsh.fltk.wait()

# When the GUI is launched, you can use the `Help->Current Options and
# Workspace' menu to see the current values of all options. To save the options
# in a file, use `File->Export->Gmsh Options', or through the api:

# gmsh.write("t3.opt");

gmsh.finalize()

