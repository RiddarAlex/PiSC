""" Code for approximating TPMS relative density """

#import gmsh
import math


pi = math.pi
sin = lambda x : math.sin(x)
cos = lambda x : math.cos(x)

#gmsh.initialize()

def createGeometryAndMesh():
    # Clear all models and create a new one
    #gmsh.clear()
    #gmsh.model.add("gyroid")

    x_res = 200   # Resolution of the 3D model. Number of points along x of the unit cell
    y_res = 200
    z_res = 200

    ## Gyroid
    #F = lambda x, y, z : cos(2*pi*(x))*sin(2*pi*(y)) + cos(2*pi*(y))*sin(2*pi*(z)) + cos(2*pi*(z))*sin(2*pi*(x))
    # Schoen-IWP
    F = lambda x, y, z : 2*(cos(2*pi*x)*cos(2*pi*y)+cos(2*pi*y)*cos(2*pi*z)+cos(2*pi*z)*cos(2*pi*x))-(cos(2*2*pi*x)+cos(2*2*pi*y)+cos(2*2*pi*z))
    C = lambda x, y, z : 0.05*pi

    oldDataIsovals = [0.157079632679490,0.184052902937584,0.211026173195678,0.237999443453772,0.264972713711866,0.291945983969961,0.318919254228055,0.345892524486149,0.372865794744243,0.399839065002337,0.426812335260431,0.453785605518526,0.480758875776620,0.507732146034714,0.534705416292808,0.561678686550902,0.588651956808997,0.615625227067091,0.642598497325185,0.669571767583279,0.750491578357562,0.777464848615656,0.804438118873750,0.831411389131844,0.858384659389938,0.885357929648033,0.912331199906127,0.939304470164221,0.966277740422315,0.993251010680409,1.02022428093850,1.04719755119660,1.07417082145469,1.10114409171279,1.12811736197088,1.37087679429373,1.39785006455182,1.42482333480992,1.45179660506801,1.47876987532610,1.50574314558420,1.66758276713276,1.88336892919752,1.91034219945561,1.93731546971371,1.96428873997180,1.99126201022989,2.01823528048799,2.04520855074608,2.07218182100418,2.18007490203655,2.20704817229465,2.50375414513368]
    newDataIsovals = [0.381143072540796,0.392935885165075,0.404728697789354,0.416521510413634,0.428314323037913,0.440107135662192,0.451899948286472,0.463692760910751,0.475485573535030,0.487278386159309,0.499071198783589,0.510864011407868,0.522656824032147,0.534449636656426,0.546242449280706,0.687756200772057,0.805684327014850,0.817477139639129,0.829269952263408,0.841062764887688,0.852855577511967,0.864648390136246,0.876441202760525,0.888234015384805,0.900026828009084,0.911819640633363,0.923612453257643,0.935405265881922,0.947198078506201,0.994369329003318,1.00616214162760,1.01795495425188,1.02974776687616,1.04154057950044,1.05333339212471,1.06512620474899,1.07691901737327,1.08871182999755,1.10050464262183,1.11229745524611,1.12409026787039,1.13588308049467,1.14767589311895,1.21843276886462,1.23022558148890,1.24201839411318,1.25381120673746,1.26560401936174,1.27739683198602,1.28918964461030,1.30098245723458,1.31277526985886,1.32456808248314,1.33636089510742,1.34815370773170,1.35994652035598,1.37173933298025,1.38353214560453,1.39532495822881,1.40711777085309,1.41891058347737,1.43070339610165,1.44249620872593,1.45428902135021,1.47787464659877,1.48966745922305,1.50146027184733,1.51325308447161,1.52504589709589,1.58400996021728,1.59580277284156,1.85524465057571,1.89062308844854,1.90241590107282,1.93779433894566,1.94958715156994,2.07930809043701,2.10289371568557]
    
    oldDataRD = []
    newDataRD = []

    datacounter = 0
    for isovalue in oldDataIsovals:
        datacounter += 1
        counter = 0
        for i in range(x_res+1):
            for j in range(y_res+1):
                for k in range(z_res+1):
                    F_value = F(i/x_res, j/y_res, k/z_res)
                    C_value = isovalue #C(i/x_res, j/y_res, k/z_res)
                    if -C_value <= F_value and F_value <= C_value:
                        #tag = i*10^6+j*10^3+k
                        #gmsh.model.geo.addPoint(0+i/x_res, 0+j/y_res, 0+k/z_res, lc)
                        counter += 1
    
        print(f'Counter is {counter}')
        print(f'Cube volume is {x_res*y_res*z_res}')
        Vol = counter/(x_res*y_res*z_res)
        print(f'Volume: {100*Vol}%. {datacounter} done and {len(oldDataIsovals)} to go')
        oldDataRD.append(100*Vol)

    datacounter = 0
    for isovalue in newDataIsovals:
        datacounter += 1
        counter = 0
        for i in range(x_res+1):
            for j in range(y_res+1):
                for k in range(z_res+1):
                    F_value = F(i/x_res, j/y_res, k/z_res)
                    C_value = isovalue #C(i/x_res, j/y_res, k/z_res)
                    if -C_value <= F_value and F_value <= C_value:
                        #tag = i*10^6+j*10^3+k
                        #gmsh.model.geo.addPoint(0+i/x_res, 0+j/y_res, 0+k/z_res, lc)
                        counter += 1
    
        print(f'Counter is {counter}')
        print(f'Cube volume is {x_res*y_res*z_res}')
        Vol = counter/(x_res*y_res*z_res)
        print(f'Volume: {100*Vol}%. {datacounter} done and {len(newDataIsovals)} to go')
        newDataRD.append(100*Vol)

    print(oldDataRD)
    print(newDataRD)

    #gmsh.model.geo.synchronize()

    #gmsh.model.mesh.generate(3)
    #gmsh.write("t3.msh")

createGeometryAndMesh()

# # Launch the GUI and handle the "check" event (recorded in the "ONELAB/Action"
# # parameter) to recreate the geometry with a new twisting angle if necessary:
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

# # When the GUI is launched, you can use the `Help->Current Options and
# # Workspace' menu to see the current values of all options. To save the options
# # in a file, use `File->Export->Gmsh Options', or through the api:

# # gmsh.write("t3.opt");

#gmsh.finalize()
