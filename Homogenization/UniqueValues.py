# This code extracts the unique values of the C-matrices and writes the average of these values to a txt-file

import numpy as np

data = np.loadtxt("Computational Homogenization/CdataIWP_new.txt", dtype='i', delimiter=' ')
# print(data)

data = data.flatten()

numberOfMatrices = 78 # Change this value to match the number of matrices in Cdata

Cmat = (np.zeros((numberOfMatrices, 36)))

for row in range(numberOfMatrices):
    Cmat[row] = data[36*row:36*(row+1)]

# Matrix is flattened. The indexes are as follows:
# C11 = [0], C12 = [1], C13 = [2]
#            C22 = [7], C23 = [8]
#                       C33 = [14]
#                                   C44 = [21]
#                                               C55 = [28]
#                                                           C66 = [35]

unique_values_average = np.zeros((numberOfMatrices, 3))

# Fill list with the average of the unique elements
for row in range(len(Cmat)):
    unique_values_average[row] = [np.mean([Cmat[row][0], Cmat[row][7], Cmat[row][14]]), np.mean([Cmat[row][1], Cmat[row][2], Cmat[row][8]]), np.mean([Cmat[row][21], Cmat[row][28], Cmat[row][35]])]

# Write in usual order
with open('DataGeneration/uniqueValuesAverageIWPNew.txt', 'a') as f:
    for row in unique_values_average:
        f.write(f"[{row[0]}, {row[1]}, {row[2]}], ")
    f.write("]")
    f.close()
# Need to edit the txt manually to add '[' in the beginning and remove ', ' at the end

# Write reverse order, if the Cdata is in reverse
# with open('DataGeneration/uniqueValuesAverageHollowCubesn300.txt', 'a') as f:
#     for i in range(numberOfMatrices-1, -1, -1):
#         f.write(f"[{unique_values_average[i][0]}, {unique_values_average[i][1]}, {unique_values_average[i][2]}], ")
#     f.write("]")
#     f.close()
# Need to edit the txt manually to add '[' in the beginning and remove ', ' at the end