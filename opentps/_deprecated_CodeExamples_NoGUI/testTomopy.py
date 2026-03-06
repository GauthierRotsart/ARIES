print('start')
from opentps.core.processing.imageSimulation.DRRToolBox import forwardProjection
from opentps.core.io.serializedObjectIO import loadDataStructure
import matplotlib.pyplot as plt

dataPath1 = "D:/ARIES/Patient_12\FDG1\dynModAndROIs.p"
patient1 = loadDataStructure(dataPath1)[0]
dynMod1 = patient1.getPatientDataOfType("Dynamic3DModel")[0]

ct = dynMod1.midp
print(type(ct))

DRR = forwardProjection(ct, 0)
print(type(DRR))

try:
    plt.figure()
    plt.imshow(DRR)
    plt.show()
except:
    pass


