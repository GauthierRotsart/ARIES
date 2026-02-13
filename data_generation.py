import os
import sys
import logging
import matplotlib.pyplot as plt
import math
import time
import copy
import pandas as pd

from tqdm import tqdm
from numpy.random import uniform as unif
from opentps.core.data.dynamicData._breathingSignals import SyntheticBreathingSignal
from opentps.core.processing.deformableDataAugmentationToolBox.generateDynamicSequencesFromModel import \
    generateDeformationListFromBreathingSignalsAndModel
from opentps.core.io.serializedObjectIO import saveSerializedObjects, loadDataStructure
from opentps.core.processing.imageProcessing.syntheticDeformation import applyBaselineShift, shrinkOrgan
from opentps.core.processing.deformableDataAugmentationToolBox import multiProcSpawnMethods
from opentps.core.processing.segmentation.segmentation3D import getBoxAroundROI
from opentps.core.processing.imageProcessing.resampler3D import resample, crop3DDataAroundBox
from opentps.core.processing.deformableDataAugmentationToolBox.modelManipFunctions import *
from opentps.core.processing.imageProcessing.imageTransform3D import translateData, rotateData
from opentps.core.data._patient import Patient

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # le premier pour l idx serie
    # le deuxieme pour l id du patient
    # le troisieme pour la regularite du signal
    # le quatrieme pour la partie du dataset
    # le cinquieme pour le modele registered (21, 31 ou 32)
    # le sixieme pour l etude
    print(f'Working with Patient {sys.argv[2]} and serie {sys.argv[1]} from study {sys.argv[6]}')
    data_augmentation = False  #P108-15
    cropping = False
    # use Z - 0 for Coronal and Z - 90 for sagittal
    # En projettant selon l axe Z, on projette dans le plan XY. Donc, le spacing de la DRR est le spacing Sx x Sy
    # de l image midp.
    projAngle = 0
    projAxis = 'Z'
    outputSize = [512, 512]

    if sys.argv[6] == "TCIA":
        studyFolder = '4D-Lung'
    elif sys.argv[6] == "FDG":
        studyFolder = 'FDGorFAZA_study'
    elif sys.argv[6] == "NO_CPAP":
        studyFolder = 'CPAP_study'
    else:
        print("Choose an appropriate study folder")
        raise NotImplementedError

    basePath = f'/Benson_DATA1/Public/ARIES/{studyFolder}/Patient_{sys.argv[2]}'
    if sys.argv[6] == "TCIA":
        patientComplement1 = '1'
        patientComplement2 = '2'
        bodyContourToUse1 = None
        bodyContourToUse2 = None
        targetContourToUse1 = 'Tumor_c00'
        targetContourToUse2 = 'Tumor_c00' if int(sys.argv[2]) != 115 else 'Tumor_ c00'
        dataPath1 = f'{basePath}/{patientComplement1}/dynModAndROIs_Patient_{sys.argv[2]}_TCIA_1.p'
        dataPath2 = f'{basePath}/{patientComplement2}/dynModAndROIs_Patient_{sys.argv[2]}_TCIA_2.p'
        dataPath_registered = f'{basePath}/{patientComplement2}/Patient_{sys.argv[2]}_registered_21.p'
    elif sys.argv[6] == "FDG":
        patientComplement1 = '1/FDG1'
        patientComplement2 = '2/FDG2'
        organsName = pd.read_excel('/linux/grotsartdehe/PatientsOrgansNames.xlsx', sheet_name=0).values
        bodyContourToUse1 = organsName[0, int(sys.argv[2])]
        bodyContourToUse2 = organsName[1, int(sys.argv[2])]
        targetContourToUse1 = organsName[3, int(sys.argv[2])]
        targetContourToUse2 = organsName[4, int(sys.argv[2])]
        dataPath1 = f'{basePath}/{patientComplement1}/dynModAndROIs_Patient_{sys.argv[2]}_FDG_1.p'
        dataPath2 = f'{basePath}/{patientComplement2}/dynModAndROIs_Patient_{sys.argv[2]}_FDG_2.p'
        dataPath_registered = f'{basePath}/{patientComplement2}/Patient_{sys.argv[2]}_registered_21.p'
    elif sys.argv[6] == "NO_CPAP":
        patientComplement1 = '1/NO_CPAP'
        patientComplement2 = '2/NO_CPAP'
        organsName = pd.read_excel('/linux/grotsartdehe/PatientsOrgansNames.xlsx', sheet_name=2).values
        bodyContourToUse1 = organsName[0, int(sys.argv[2])]
        bodyContourToUse2 = organsName[1, int(sys.argv[2])]
        targetContourToUse1 = organsName[2, int(sys.argv[2])]
        targetContourToUse2 = organsName[3, int(sys.argv[2])]
        dataPath1 = f'{basePath}/{patientComplement1}/dynModAndROIs_Patient_{sys.argv[2]}_NO_CPAP_1.p'
        dataPath2 = f'{basePath}/{patientComplement2}/dynModAndROIs_Patient_{sys.argv[2]}_NO_CPAP_2.p'
        dataPath_registered = f'{basePath}/{patientComplement2}/Patient_{sys.argv[2]}_registered.p'
    else:
        print("Choose an appropriate study")
        raise NotImplementedError

    # sequence duration, sampling and signal's regularity
    regularityIndex = int(sys.argv[3])
    samplingFrequency = 5
    GPUNumber = 0

    if sys.argv[4] == "train":
        if data_augmentation is True:
            datasetPart = 'training_images_augmented_augmented'
        else:
            datasetPart = 'training_images'
        movingModel = "dynMod1"
        sequenceDurationInSecs = 300
        print("Dataset part: training")
    elif sys.argv[4] == "train_T2":
        datasetPart = 'training_images_T2'
        movingModel = "dynMod2"
        sequenceDurationInSecs = 300
        print("Dataset part: training T2")
    elif sys.argv[4] == "test":
        datasetPart = f'test_images_{sys.argv[5]}'
        movingModel = "dynMod2"
        sequenceDurationInSecs = 40
        print("Dataset part: test")
    else:
        print("Error in the dataset part")
        raise NotImplementedError

    try:
        import cupy
        import cupyx

        cupy.cuda.Device(GPUNumber).use()
    except:
        print('cupy not found.')

    tryGPU = True

    # breathing signal parameters
    amplitude = 'model'
    breathingMotionDirection = 'Z'
    breathingPeriod = 4
    meanNoise = 0
    samplingPeriod = 1 / samplingFrequency

    # multiProcessing
    maxMultiProcUse = int(sys.argv[7])
    subSequenceSize = maxMultiProcUse

    if regularityIndex == 1:
        regularityFolder = 'Regular'
        print("Regularity of the signal: regular")
    elif regularityIndex == 2:
        regularityFolder = 'Middle'
        print("Regularity of the signal: middle")
    elif regularityIndex == 3:
        regularityFolder = 'Irregular'
        print("Regularity of the signal: irregular")
    else:
        print("Regularity index error. Choose an index between 1 and 3.")
        raise NotImplementedError

    startTime = time.time()
    totalIeFTime = 0
    totalIaFTime = 0
    totalSavingTime = 0
    totalLoadTime = 0

    # Start the script ---------------------------------
    curLoadStartTime = time.time()
    patient1 = loadDataStructure(dataPath1)[0]
    dynMod1 = patient1.getPatientDataOfType("Dynamic3DModel")[0]
    rtStruct1 = patient1.getPatientDataOfType("RTStruct")[0]
    print('Available ROIs')
    rtStruct1.print_ROINames()

    patient2 = loadDataStructure(dataPath2)[0]
    dynMod2 = patient2.getPatientDataOfType("Dynamic3DModel")[0]
    rtStruct2 = patient2.getPatientDataOfType("RTStruct")[0]
    print(dynMod2.midp.imageArray.shape)

    patient_registred = loadDataStructure(dataPath_registered)[0]
    dynMod_registered = patient_registred.getPatientDataOfType("Dynamic3DModel")[0]
    print('Available ROIs')
    rtStruct2.print_ROINames()
    curLoadStopTime = time.time()
    print('Time for loading model: ', curLoadStopTime - curLoadStartTime)
    totalLoadTime += curLoadStopTime - curLoadStartTime

    # Get the ROI and mask on which we want to apply the motion signal
    gtvContour1 = rtStruct1.getContourByName(targetContourToUse1)
    GTVMask1 = gtvContour1.getBinaryMask(origin=dynMod1.midp.origin, gridSize=dynMod1.midp.gridSize,
                                         spacing=dynMod1.midp.spacing)

    gtvContour2 = rtStruct2.getContourByName(targetContourToUse2)
    GTVMask2 = gtvContour2.getBinaryMask(origin=dynMod2.midp.origin, gridSize=dynMod2.midp.gridSize,
                                         spacing=dynMod2.midp.spacing)
    GTVMask_registered = dynMod_registered.maskList

    # Get the 3D center of mass of this ROI
    gtvCenterOfMass1 = gtvContour1.getCenterOfMass(origin=dynMod1.midp.origin, gridSize=dynMod1.midp.gridSize,
                                                   spacing=dynMod1.midp.spacing)  # coord du scanner en mm
    GTVCenterOfMassInVoxels1 = getVoxelIndexFromPosition(gtvCenterOfMass1, dynMod1.midp)  # coord en pixel

    gtvCenterOfMass2 = gtvContour2.getCenterOfMass(origin=dynMod2.midp.origin, gridSize=dynMod2.midp.gridSize,
                                                   spacing=dynMod2.midp.spacing)  # coord du scanner en mm
    GTVCenterOfMassInVoxels2 = getVoxelIndexFromPosition(gtvCenterOfMass2, dynMod2.midp)  # coord en pixel

    gtvCenterOfMass_registered = GTVMask_registered.centerOfMass
    GTVCenterOfMassInVoxels_registered = getVoxelIndexFromPosition(gtvCenterOfMass_registered,
                                                                   dynMod_registered.midp)  # coord en pixel
    print('COM of the model 1 in mm:', gtvCenterOfMass1)
    print('COM of the model 1 in voxel:', GTVCenterOfMassInVoxels1)
    print('COM of the model 2 in mm:', gtvCenterOfMass2)
    print('COM of the model 2 in voxel:', GTVCenterOfMassInVoxels2)
    print('COM of the model registered in mm:', gtvCenterOfMass_registered)
    print('COM of the model registered in voxel:', GTVCenterOfMassInVoxels_registered)

    if sys.argv[4] == "train" or (sys.argv[4] == "test" and int(sys.argv[5]) == 11):  # dynMod1
        dynModCopy = copy.deepcopy(dynMod1)  # moving dynMod
        GTVMaskCopy = copy.deepcopy(GTVMask1)  # moving GTVMask
        dynModSITK = copy.deepcopy(dynMod_registered)  # fixed dynMod
        GTVMaskSITK = copy.deepcopy(GTVMask_registered)  # fixed GTVMask
        savingPath = f'/Benson_DATA1/grotsart/{studyFolder}/Patient_{sys.argv[2]}/1/{regularityFolder}/{datasetPart}/' \
                     f'serie{sys.argv[1]}'
    elif sys.argv[4] == "train_T2" or (sys.argv[4] == "test" and int(sys.argv[5]) == 21):  # dynMod2
        dynModCopy = copy.deepcopy(dynMod_registered)  # moving dynMod
        GTVMaskCopy = copy.deepcopy(GTVMask_registered)  # moving GTVMask
        dynModSITK = copy.deepcopy(dynMod1)  # fixed dynMod
        GTVMaskSITK = copy.deepcopy(GTVMask1)  # fixed GTVMask
        savingPath = f'/Benson_DATA1/grotsart/{studyFolder}/Patient_{sys.argv[2]}/2/{regularityFolder}/{datasetPart}/' \
                     f'serie{sys.argv[1]}'
    else:
        print(f"Not implemented config: {sys.argv[4]} and {sys.argv[5]}", )
        raise NotImplementedError

    if not os.path.exists(savingPath):
        os.umask(0)
        os.makedirs(savingPath)  # Create a new directory because it does not exist
        print("New directory created to save the data: ", savingPath)

    if cropping is True:  # Define the cropping box
        # On definit la cropping box avant les déformations interfractions pour cropper toujours de la meme maniere
        # et pouvoir observer les translations
        # Cropping box identique pour training et test set
        gtvBox = getBoxAroundROI(GTVMask1)
        bodyContour = rtStruct1.getContourByName(bodyContourToUse1)
        bodyMask = bodyContour.getBinaryMask(origin=dynMod1.midp.origin, gridSize=dynMod1.midp.gridSize,
                                             spacing=dynMod1.midp.spacing)
        bodyBox = getBoxAroundROI(bodyMask)

        croppingContoursUsedXYZ = [targetContourToUse1, bodyContourToUse1, targetContourToUse1]
        croppingBox = [[], [], []]
        for i in range(3):
            if croppingContoursUsedXYZ[i] == bodyContourToUse1:
                croppingBox[i] = bodyBox[i]
            elif croppingContoursUsedXYZ[i] == targetContourToUse1:
                croppingBox[i] = gtvBox[i]

    # interfraction changes parameters
    if (sys.argv[4] == "train" or sys.argv[4] == "train_T2") and (data_augmentation is True):
        baselineShift = [unif(-5, 5), unif(-5, 5), unif(-5, 5)]
        translation = [0, 0, 0]  # [unif(-3, 3), unif(-3, 3), unif(-3, 3)]
        rotation = np.array([0, 0, 0])  # np.array([unif(-5, 5), unif(-5, 5), unif(-5, 5)])
        rotationInRad = (rotation * 2 * math.pi) / 360
        shrinkValue = unif(0, 3)
        shrinkSize = [shrinkValue + np.random.normal(0, 0.5), shrinkValue + np.random.normal(0, 0.5),
                      shrinkValue + np.random.normal(0, 0.5)]
        #shrinkSize = [0, 0, 0]

        GTVMaskCopy.imageArray = GTVMaskCopy.imageArray.astype(float)

        # Translation
        translateData(dynModCopy, translationInMM=translation, outputBox='same', tryGPU=tryGPU, interpOrder=1,
                      mode='nearest', fillValue=-1000)
        translateData(GTVMaskCopy, translationInMM=translation, outputBox='same', tryGPU=tryGPU, interpOrder=1,
                      mode='nearest', fillValue=0)
        cupy._default_memory_pool.free_all_blocks()

        # Rotation
        rotateData(dynModCopy, rotAnglesInDeg=rotation, rotCenter='imgCenter', outputBox='same', tryGPU=tryGPU,
                   interpOrder=1, mode='nearest', fillValue=-1000)
        rotateData(GTVMaskCopy, rotAnglesInDeg=rotation, rotCenter='imgCenter', outputBox='same', tryGPU=tryGPU,
                   interpOrder=1, mode='nearest', fillValue=0)
        cupy._default_memory_pool.free_all_blocks()

        # Baseline shift
        dynModCopy, GTVMaskCopy = applyBaselineShift(dynModCopy, GTVMaskCopy, baselineShift, tryGPU=tryGPU)
        cupy._default_memory_pool.free_all_blocks()

        # Shrink
        GTVMaskCopy.imageArray[GTVMaskCopy.imageArray > 0.5] = 1
        GTVMaskCopy.imageArray[GTVMaskCopy.imageArray <= 0.5] = 0
        GTVMaskCopy.imageArray = GTVMaskCopy.imageArray.astype(bool)

        dynModCopy, GTVMaskCopy = shrinkOrgan(dynModCopy, GTVMaskCopy, shrinkSize=shrinkSize, tryGPU=False)
        cupy._default_memory_pool.free_all_blocks()
    else:
        rotationInRad = np.array([0, 0, 0])

    print('Moving center of mass in mm:', GTVMaskCopy.centerOfMass)
    print('Moving center of mass in voxels:',
          getVoxelIndexFromPosition(GTVMaskCopy.centerOfMass, dynModCopy.midp))
    print('Fixed center of mass in mm:', GTVMaskSITK.centerOfMass)
    print('Fixed center of mass in voxels:',
          getVoxelIndexFromPosition(GTVMaskSITK.centerOfMass, dynModSITK.midp))

    if cropping is True:
        # crop the model data using the box
        # Pour gagner du temps lors des deformations intra-fractions
        marginInMM = [50, 0, 100]
        crop3DDataAroundBox(dynModCopy, croppingBox, marginInMM=marginInMM)
        crop3DDataAroundBox(GTVMaskCopy, croppingBox, marginInMM=marginInMM)
        crop3DDataAroundBox(dynModSITK, croppingBox, marginInMM=marginInMM)
        crop3DDataAroundBox(GTVMaskSITK, croppingBox, marginInMM=marginInMM)

        # if you want to see the crop in the GUI you can save the data in cropped version
        # Create a patient and give it the patient name
        patient = Patient()
        patient.name = f'Patient_{sys.argv[2]}'
        # Add the model and rtStruct to the patient
        patient.appendPatientData(dynModCopy)
        if sys.argv[4] == "train" or (sys.argv[4] == "test" and int(sys.argv[5]) == 11):
            patient.appendPatientData(rtStruct1)
        elif sys.argv[4] == "train_T2" or (sys.argv[4] == "test" and int(sys.argv[5]) == 21):
            patient.appendPatientData(rtStruct1)
        else:
            raise NotImplementedError
        saveSerializedObjects(patient, f'{savingPath}/Patient_{sys.argv[2]}_Cropped_Model_And_ROIs')

    original_spacing_moving = np.array(dynModCopy.midp.spacing)  # [sx, sy, sz]
    original_spacing_fixed = np.array(dynModSITK.midp.spacing)  # [sx, sy, sz]
    original_shape_moving = np.array(dynModCopy.midp.imageArray.shape)  # [512, 512, 64]
    original_shape_fixed = np.array(dynModSITK.midp.imageArray.shape)  # [512, 512, 64]
    target_shape = np.array([128, 128, 128])
    new_spacing_moving = (original_spacing_moving * original_shape_moving) / target_shape
    new_spacing_fixed = (original_spacing_fixed * original_shape_fixed) / target_shape

    # on resample
    resample(dynModSITK,
             spacing=new_spacing_fixed,
             inPlace=True)
    resample(GTVMaskSITK,
             spacing=new_spacing_fixed,
             inPlace=True)
    resample(dynModCopy,
             spacing=new_spacing_moving,
             inPlace=True)
    resample(GTVMaskCopy,
             spacing=new_spacing_moving,
             inPlace=True)

    gtvCenterOfMass = GTVMaskCopy.centerOfMass  # coord du scanner en mm
    GTVCenterOfMassInVoxels = getVoxelIndexFromPosition(gtvCenterOfMass, dynModCopy.midp)  # coord en pixel
    print('Center of masse should be the same without cropping', gtvCenterOfMass)

    # Signal creation
    if regularityIndex == 1:
        varianceNoise = np.random.uniform(0.5, 1.5)
        coeffMin = 0.10
        coeffMax = 0.15
        meanEvent = 1 / 60
        meanEventApnea = 0 / 120
    elif regularityIndex == 2:
        varianceNoise = np.random.uniform(1.5, 2.5)
        coeffMin = 0.10
        coeffMax = 0.45
        meanEvent = 1 / 30
        meanEventApnea = 0 / 120
    elif regularityIndex == 3:
        varianceNoise = np.random.uniform(2.5, 3.5)
        coeffMin = 0.10
        coeffMax = 0.45
        meanEvent = 1 / 20
        meanEventApnea = 1 / 120
    else:
        print("Regularity index error. Choose an index between 1 and 3.")
        raise NotImplementedError

    if amplitude == 'model':
        # to get amplitude from model !!! it takes some time because 10 displacement fields must be computed just for
        # this
        modelValues_Z = getAverageModelValuesAroundPosition(gtvCenterOfMass, dynModCopy, dimensionUsed='Z')
        minData_Z = np.min(modelValues_Z)
        maxData_Z = np.max(modelValues_Z)
        amplitude_Z = maxData_Z - minData_Z
        print('Amplitude of deformation at ROI center of mass', amplitude_Z)
    else:
        """# if there are inter-fraction rotations, the amplitude is adapted such as its value is applied in the rotated 
        # direction
        if breathingMotionDirection == 'Z':
            amplitude *= math.cos(rotationInRad[0]) * math.cos(rotationInRad[1])
        elif breathingMotionDirection == 'X':
            amplitude *= math.cos(rotationInRad[1]) * math.cos(rotationInRad[2])
        elif breathingMotionDirection == 'Y':
            amplitude *= math.cos(rotationInRad[0]) * math.cos(rotationInRad[2])
        else: 
            print("Not a direction")
            raise NotImplementedError
        print('Amplitude to apply in', breathingMotionDirection, 'after rotations :', amplitude)"""
        raise NotImplementedError

    newSignal = SyntheticBreathingSignal(amplitude=amplitude_Z,
                                         breathingPeriod=breathingPeriod,
                                         meanNoise=meanNoise,
                                         varianceNoise=varianceNoise,
                                         samplingPeriod=samplingPeriod,
                                         simulationTime=sequenceDurationInSecs,
                                         coeffMin=coeffMin,
                                         coeffMax=coeffMax,
                                         meanEvent=meanEvent,
                                         meanEventApnea=meanEventApnea)

    newSignal.generate1DBreathingSignal()
    pointList = [gtvCenterOfMass]
    pointVoxelList = [GTVCenterOfMassInVoxels]
    signalList = [newSignal.breathingSignal]
    saveSerializedObjects([signalList, pointList],
                          f'{savingPath}/Patient_{sys.argv[2]}_ROI_And_Signal_object_serie_{sys.argv[1]}')
    cupy._default_memory_pool.free_all_blocks()

    # to show signals and ROIs
    # Show the moving image
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure(figsize=(12, 6))
    signalAx = plt.subplot(2, 1, 2)
    for pointIndex, point in enumerate(pointList):
        ax = plt.subplot(2, 2 * len(pointList), 2 * pointIndex + 1)
        ax.set_title('Slice Y:' + str(pointVoxelList[pointIndex][1]))
        ax.imshow(np.rot90(dynModCopy.midp.imageArray[:, pointVoxelList[pointIndex][1], :]))
        ax.imshow(np.rot90(GTVMaskCopy.imageArray[:, pointVoxelList[pointIndex][1], :]), alpha=0.3)
        ax.scatter([pointVoxelList[pointIndex][0]],
                   [dynModCopy.midp.imageArray.shape[2] - pointVoxelList[pointIndex][2]],
                   c=colors[pointIndex], marker="x", s=100)
        """
        ax2 = plt.subplot(2, 2 * len(pointList), 2 * pointIndex + 2)
        ax2.set_title('Slice Z:' + str(pointVoxelList[pointIndex][2]))
        ax2.imshow(np.rot90(dynModCopy.midp.imageArray[:, :, str(pointVoxelList[pointIndex][2])], 3))
        ax2.imshow(np.rot90(GTVMaskCopy.imageArray[:, :, str(pointVoxelList[pointIndex][2])], 3), alpha=0.3)
        ax2.scatter([pointVoxelList[pointIndex][0]], [pointVoxelList[pointIndex][1]],
                   c=colors[pointIndex], marker="x", s=100)
        """
        signalAx.plot(newSignal.timestamps / 1000, signalList[pointIndex], c=colors[pointIndex])

    signalAx.set_xlabel('Time (s)')
    signalAx.set_ylabel('Deformation amplitude in Z direction (mm)')
    plt.savefig(f'{savingPath}/Patient_{sys.argv[2]}_moving_fig_serie_{sys.argv[1]}.pdf', dpi=300)
    # plt.show()
    plt.close()

    # Show the fixed image
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure(figsize=(12, 6))
    signalAx = plt.subplot(2, 1, 2)
    for pointIndex, point in enumerate(pointList):
        ax = plt.subplot(2, 2 * len(pointList), 2 * pointIndex + 1)
        ax.set_title('Slice Y:' + str(pointVoxelList[pointIndex][1]))
        ax.imshow(np.rot90(dynModSITK.midp.imageArray[:, pointVoxelList[pointIndex][1], :]))
        ax.imshow(np.rot90(GTVMaskSITK.imageArray[:, pointVoxelList[pointIndex][1], :]), alpha=0.3)
        ax.scatter([pointVoxelList[pointIndex][0]],
                   [dynModSITK.midp.imageArray.shape[2] - pointVoxelList[pointIndex][2]],
                   c=colors[pointIndex], marker="x", s=100)
        """
        ax2 = plt.subplot(2, 2 * len(pointList), 2 * pointIndex + 2)
        ax2.set_title('Slice Z:' + str(str(pointVoxelList[pointIndex][2])))
        ax2.imshow(np.rot90(dynModSITK.midp.imageArray[:, :, str(pointVoxelList[pointIndex][2])], 3))
        ax2.imshow(np.rot90(GTVMaskSITK.imageArray[:, :, str(pointVoxelList[pointIndex][2])], 3), alpha=0.3)
        ax2.scatter([pointVoxelList[pointIndex][0]], [pointVoxelList[pointIndex][1]],
                    c=colors[pointIndex], marker="x", s=100)
        """
        signalAx.plot(newSignal.timestamps / 1000, signalList[pointIndex], c=colors[pointIndex])

    signalAx.set_xlabel('Time (s)')
    signalAx.set_ylabel('Deformation amplitude in Z direction (mm)')
    plt.savefig(f'{savingPath}/Patient_{sys.argv[2]}_fixed_fig_serie_{sys.argv[1]}.pdf', dpi=300)
    # plt.show()
    plt.close()

    dynModSITK = resample(dynModSITK, spacing=dynModCopy.midp.spacing, origin=dynModCopy.midp.origin,
                          gridSize=dynModCopy.midp.gridSize, fillValue=-1000)
    midpSITK = dynModSITK.midp  # GTVMaskSITK
    midpCUPY = dynModCopy.midp  # GTVMaskCopy
    differenceMidp = midpCUPY.imageArray[:, pointVoxelList[pointIndex][1], :] - \
                     midpSITK.imageArray[:, pointVoxelList[pointIndex][1], ]
    plt.figure(figsize=(15, 8))
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    ax[0].imshow(midpCUPY.imageArray[:, pointVoxelList[pointIndex][1], :].T)
    ax[0].set_title("Moving image")
    ax[1].imshow(midpSITK.imageArray[:, pointVoxelList[pointIndex][1], :].T)
    ax[1].set_title("Fixed image")
    ax[2].imshow(differenceMidp.T)
    ax[2].set_title("Difference between the two images")
    plt.title(f'Patient_{sys.argv[2]}')
    plt.savefig(f'{savingPath}/Patient_{sys.argv[2]}_difference.pdf', dpi=300)
    # plt.show()
    plt.close()

    sequenceSize = newSignal.breathingSignal.shape[0]
    print(f'Sequence size = {sequenceSize}, split by stack of {subSequenceSize}. Multiprocessing = {maxMultiProcUse}')

    subSequencesIndexes = [subSequenceSize * i for i in range(math.ceil(sequenceSize / subSequenceSize))]
    subSequencesIndexes.append(sequenceSize)
    print('Sub sequences indexes', subSequencesIndexes)

    resultList = []

    if subSequenceSize > maxMultiProcUse:  # re-adjust the subSequenceSize since this will be done in multi processing
        subSequenceSize = maxMultiProcUse
        subSequencesIndexes = [subSequenceSize * i for i in range(math.ceil(sequenceSize / subSequenceSize))]
        subSequencesIndexes.append(sequenceSize)

    with tqdm(total=samplingFrequency * sequenceDurationInSecs, unit="img", desc=f"Image") as pbar:
        for i in range(len(subSequencesIndexes) - 1):
            deformationList = generateDeformationListFromBreathingSignalsAndModel(dynModCopy,
                                                                                  signalList,
                                                                                  pointList,
                                                                                  signalIdxUsed=[
                                                                                      subSequencesIndexes[i],
                                                                                      subSequencesIndexes[
                                                                                          i + 1]],
                                                                                  dimensionUsed='Z',
                                                                                  outputType=np.float32)

            deformedImgMaskAnd3DCOMList = multiProcSpawnMethods.multiProcDeform(deformationList, dynModCopy,
                                                                                GTVMaskCopy,
                                                                                ncore=maxMultiProcUse,
                                                                                GPUNumber=GPUNumber)
            if i == 0:
                plt.figure()
                plt.imshow(deformedImgMaskAnd3DCOMList[-1][0].imageArray[:, :, 20])
                plt.imshow(deformedImgMaskAnd3DCOMList[-1][1].imageArray[:, :, 20], alpha=0.5)
                plt.savefig(f'{savingPath}/Patient_{sys.argv[2]}_Result_Deform_serie_{sys.argv[1]}.pdf', dpi=300)
                plt.close()

            # print('Start multi process DRRs with', len(deformationList), 'pairs of image-mask')
            # projectionResults = []
            # projectionResults += multiProcDRRs(deformedImgMaskAnd3DCOMList, projAngle, projAxis, outputSize, ncore=maxMultiProcUse)
            projectionResults = multiProcSpawnMethods.multiProcDRRs(deformedImgMaskAnd3DCOMList, projAngle,
                                                                    projAxis, outputSize, ncore=maxMultiProcUse)
            if i == 0:
                plt.figure()
                plt.imshow(projectionResults[-1][0], cmap='Greys')
                # plt.imshow(projectionResults[-1][1], alpha=0.5)
                plt.savefig(f'{savingPath}/Patient_{sys.argv[2]}_Result_DRR_serie_{sys.argv[1]}.pdf', dpi=300)
                # plt.show()
                plt.close()

            # add 3D center of mass in scanner coordinates to the result lists
            for imgIndex in range(len(projectionResults)):
                projectionResults[imgIndex].append(deformedImgMaskAnd3DCOMList[imgIndex][2])

            resultList += projectionResults
            # print('ResultList lenght', len(resultList))
            pbar.update(len(projectionResults))
    serieSavingPath = f'{savingPath}/Patient_{sys.argv[2]}_{sequenceSize}_DRRMasksAndCOM_serie_{sys.argv[1]}'
    saveSerializedObjects(resultList, serieSavingPath)
    print('Temps d execution total: ', np.round(time.time() - startTime, 2) / 60, 'minutes')

# end = time.time()
# print('Temps d execution', end-start)
