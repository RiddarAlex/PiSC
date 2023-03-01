""" Code that combines interface surface labels in .mesh for IWP """
# Disclaimer: This code was not used during the data generation phase, but should be more intuitive than replace_label.py #
# combineSurfaceLabelsNew(filename)

def getCoordinates(lst):
    while '' in lst:
      lst.remove('')
    return lst

def combineSurfaceLabelsNew(filename):
    
    listOfLines = []
    path = "C:\\Users\\Baran\\"
    pathFile = path + filename + '.mesh'
    with open(pathFile, "r") as reader:
      for line in reader.readlines():
        listOfLines.append(line.split(' '))
  
    # Find "Triangles" and "Tetrahedra" in the file
    triang_index = listOfLines.index(['', 'Triangles\n'])
    tetra_index = listOfLines.index(['', 'Tetrahedra\n'])
    
    # Scrolla till "Edges"
    edge_index = listOfLines.index(['', 'Edges\n'])
    _surfTag = listOfLines[edge_index-1][-1]
    surfTag = int(_surfTag[:-1])

    # Loopa baklänges och kolla vilken randkoordinat varje label har.
    a = edge_index-1
    while surfTag == 1:
      a -= 1
    while surfTag > 3:
      #xCoord = float(listOfLines[a][-28])
      xCoord = float(getCoordinates(listOfLines[a])[0])
      yCoord = float(getCoordinates(listOfLines[a])[1])
      zCoord = float(getCoordinates(listOfLines[a])[2])

      if round(zCoord, 3) == -0.5:
        b = tetra_index
        while int(listOfLines[b-1][4]) != surfTag:
          b -= 1
        while int(listOfLines[b-1][4]) == surfTag:
          listOfLines[b-1][4] = '100\n'
          b -= 1
      elif round(zCoord, 3) == 0.5:
        b = tetra_index
        while int(listOfLines[b-1][4]) != surfTag:
          b -= 1
        while int(listOfLines[b-1][4]) == surfTag:
          listOfLines[b-1][4] = '110\n'
          b -= 1
      elif round(yCoord, 3) == -0.5:
        b = tetra_index
        while int(listOfLines[b-1][4]) != surfTag:
          b -= 1
        while int(listOfLines[b-1][4]) == surfTag:
          listOfLines[b-1][4] = '120\n'
          b -= 1
      elif round(yCoord, 3) == 0.5:
        b = tetra_index
        while int(listOfLines[b-1][4]) != surfTag:
          b -= 1
        while int(listOfLines[b-1][4]) == surfTag:
          listOfLines[b-1][4] = '130\n'
          b -= 1
      elif round(xCoord, 3) == -0.5:
        b = tetra_index
        while int(listOfLines[b-1][4]) != surfTag:
          b -= 1
        while int(listOfLines[b-1][4]) == surfTag:
          listOfLines[b-1][4] = '140\n'
          b -= 1
      elif round(xCoord, 3) == 0.5:
        b = tetra_index
        while int(listOfLines[b-1][4]) != surfTag:
          b -= 1
        while int(listOfLines[b-1][4]) == surfTag:
          listOfLines[b-1][4] = '150\n'
          b -= 1
      else:
        pass
      
      surfTag -= 1
      
      while int(listOfLines[a][-1]) != surfTag:
        a -= 1
      
    for i in range(len(listOfLines)):
      listOfLines[i] = ' '.join(listOfLines[i])

    # Save the file
    processedFilename = filename + 'processed'
    pathName = "C:\\Users\\Baran\\" + processedFilename + ".mesh"
    with open(pathName, "w") as f:
      for i in listOfLines:
        f.write(i)

    print("Job's done!" + filename)



combineSurfaceLabelsNew('IWPMatlab0.4')
