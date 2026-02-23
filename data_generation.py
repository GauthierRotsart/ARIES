import sys  # noqa: E402
import os  # noqa: E402

# DO NOT MODIFY
opentps_core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'opentps', 'opentps_core'))
if opentps_core_path not in sys.path:
    sys.path.insert(0, opentps_core_path)

import matplotlib.pyplot as plt  # noqa: E402
import math  # noqa: E402
import copy  # noqa: E402
import logging  # noqa: E402
logging.getLogger("opentps").setLevel(logging.WARNING)

from tqdm import tqdm  # noqa: E402
from numpy.random import uniform as unif  # noqa: E402
from opentps.opentps_core.opentps.core.data.dynamicData._breathingSignals import SyntheticBreathingSignal  # noqa: E402
from opentps.opentps_core.opentps.core.processing.deformableDataAugmentationToolBox.generateDynamicSequencesFromModel import \
    generateDeformationListFromBreathingSignalsAndModel  # noqa: E402
from opentps.opentps_core.opentps.core.io.serializedObjectIO import saveSerializedObjects  # noqa: E402
from opentps.opentps_core.opentps.core.processing.deformableDataAugmentationToolBox import multiProcSpawnMethods  # noqa: E402
from opentps.opentps_core.opentps.core.processing.imageProcessing.resampler3D import resample, crop3DDataAroundBox  # noqa: E402
from opentps.opentps_core.opentps.core.processing.imageSimulation import multiProcForkMethods  # noqa: E402
from opentps.opentps_core.opentps.core.processing.deformableDataAugmentationToolBox.modelManipFunctions import *  # noqa: E402
from opentps.opentps_core.opentps.core.data._patient import Patient  # noqa: E402
from opentps.opentps_core.opentps.core.processing.imageProcessing.imageTransform3D import getVoxelIndexFromPosition  # noqa: E402
from utils import get_patient_data, get_cropping_box, transform_patient, registration  # noqa: E402

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    patient_id = 12
    serie = 0

    data_augmentation = False
    cropping = True
    marginInMM = [50, 0, 100]
    train_bool = True
    test_bool = False

    # contours names
    gtv_contour_name_T1 = 'gtv t'#'Tumor_c00'
    gtv_contour_name_T2 = 'gtv t'#'Tumor_c00'
    if cropping is True:
        body_contour_T1 = 'patient'
        body_contour_T2 = 'external'
    else:
        body_contour_T1 = None
        body_contour_T2 = None

    # DRR images generation
    # use Z - 0 for Coronal axis projection and Z - 90 for sagittal axis projection
    projAxis = 'Z'
    projAngle = 0
    outputSize = [512, 512]  # size of the DRR

    # data augmentation parameters
    baselineShift = [0, 0, 0]  # [unif(-5, 5), unif(-5, 5), unif(-5, 5)]
    translation = [3, 3, 3] # [unif(-5, 5), unif(-5, 5), unif(-5, 5)]
    rotation = np.array([0, 0, 0])  # np.array([unif(-5, 5), unif(-5, 5), unif(-5, 5)])
    rotationInRad = (rotation * 2 * math.pi) / 360
    shrinkValue = unif(0, 3)
    shrinkSize = [0, 0, 0]  # [shrinkValue + np.random.normal(0, 0.5), shrinkValue + np.random.normal(0, 0.5), shrinkValue + np.random.normal(0, 0.5)]

    GPUNumber = 0
    maxMultiProcUse = 5  # number of cores
    tryGPU = False
    try:
        import cupy
        import cupyx

        cupy.cuda.Device(GPUNumber).use()
    except:
        print('cupy not found.')

    assert (train_bool is True and test_bool is False) or (train_bool is False and test_bool is True)

    print(f'Working with Patient {patient_id} and serie {serie} from study TCIA')
    data_path = f'./DATA/4D-Lung/Patient_{patient_id}'
    path1 = f'{data_path}/T1/dynModAndROIs_Patient_{patient_id}_FDG_1.p'
    path2 = f'{data_path}/T2/dynModAndROIs_Patient_{patient_id}_FDG_2.p'

    # breathing signal parameters
    amplitude = 'model'
    breathingMotionDirection = 'Z'
    breathingPeriod = 4
    meanNoise = 0
    samplingFrequency = 5
    samplingPeriod = 1 / samplingFrequency
    regularityIndex = 1

    # Start the script ---------------------------------
    if train_bool is True:
        patient_T1 = get_patient_data(patient_path=path1,
                                      roi_contour_name=gtv_contour_name_T1)
        patient_data = copy.deepcopy(patient_T1)
        sequenceDurationInSecs = 1
        saving_path = f'./DATA/4D-Lung/Patient_{patient_id}/training_set'
    else:
        patient_T1 = get_patient_data(patient_path=path1,
                                      roi_contour_name=gtv_contour_name_T1)
        patient_T2 = get_patient_data(patient_path=path2,
                                      roi_contour_name=gtv_contour_name_T2)
        patient_registered = registration(patient=patient_id,
                                          patient_T1=patient_T1,
                                          patient_T2=patient_T2)
        patient_data = copy.deepcopy(patient_registered)
        sequenceDurationInSecs = 20
        saving_path = f'./DATA/4D-Lung/Patient_{patient_id}/test_set'

    if not os.path.exists(saving_path):
        os.umask(0)
        os.makedirs(saving_path)  # Create a new directory because it does not exist
        print("New directory created to save the data: ", saving_path)

    if cropping is True:  # Define the cropping box
        # The cropping box is only based on patient T1
        # The cropping box is defined before data augmentation to always crop in the same way
        # The cropping box is the same for both the training set and the test set
        cropping_box = get_cropping_box(model=patient_data["model"],
                                        mask=patient_data["mask"],
                                        rtstruct=patient_data["rtstruct"],
                                        body_contour_name=body_contour_T1,
                                        roi_contour_name=gtv_contour_name_T1)

        print('Cropping completed')

    # interfraction changes parameters
    if train_bool is True and data_augmentation is True:
        transform_patient(patient=patient_data,
                          translation=translation,
                          rotation=rotation,
                          baseline_shift=baselineShift,
                          shrink_size=shrinkSize,
                          tryGPU=tryGPU,
                          GPUNumber=GPUNumber)

        print('Data augmentation completed')

    if cropping is True:
        # crop the model data using the cropping box
        crop3DDataAroundBox(patient_data["model"], cropping_box, marginInMM=marginInMM)
        crop3DDataAroundBox(patient_data["mask"], cropping_box, marginInMM=marginInMM)
        patient_data["com_mm"] = patient_data["mask"].centerOfMass
        patient_data["com_voxel"] = getVoxelIndexFromPosition(position=patient_data["com_mm"],
                                                              image3D=patient_data["model"].midp)

        # if you want to see the crop in the GUI you can save the data in cropped version
        # Create a patient and give it the patient name
        patient = Patient()
        patient.name = f'Patient_{patient_id}_cropped'
        # Add the model and rtStruct to the patient
        patient.appendPatientData(patient_data["model"])
        patient.appendPatientData(patient_data["rtstruct"])
        saveSerializedObjects(patient, f'{saving_path}/Patient_{patient_id}_Cropped_Model_And_ROIs')

    original_spacing_moving = np.array(patient_data["model"].midp.spacing)  # [sx, sy, sz]
    original_shape_moving = np.array(patient_data["model"].midp.imageArray.shape)  # [512, 512, 64]
    target_shape = np.array([128, 128, 128])
    new_spacing_moving = (original_spacing_moving * original_shape_moving) / target_shape

    # on resample
    resample(patient_data["model"],
             spacing=new_spacing_moving,
             inPlace=True)
    resample(patient_data["mask"],
             spacing=new_spacing_moving,
             inPlace=True)
    patient_data["com_mm"] = patient_data["mask"].centerOfMass
    patient_data["com_voxel"] = getVoxelIndexFromPosition(position=patient_data["com_mm"],
                                                          image3D=patient_data["model"].midp)

    # Signal creation
    # DO NOT MODIFY
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

    modelValues_Z = getAverageModelValuesAroundPosition(position=patient_data["com_mm"],
                                                        model=patient_data["model"],
                                                        dimensionUsed='Z')
    minData_Z = np.min(modelValues_Z)
    maxData_Z = np.max(modelValues_Z)
    amplitude_Z = maxData_Z - minData_Z
    print('Amplitude of deformation at ROI center of mass', amplitude_Z)

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
    pointList = [patient_data["com_mm"]]
    pointVoxelList = [patient_data["com_voxel"]]
    signalList = [newSignal.breathingSignal]
    # to show signals and ROIs
    # Show the moving image
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure(figsize=(12, 6))
    signalAx = plt.subplot(2, 1, 2)
    for pointIndex, point in enumerate(pointList):
        ax = plt.subplot(2, 2 * len(pointList), 2 * pointIndex + 1)
        ax.set_title('Slice Y:' + str(pointVoxelList[pointIndex][1]))
        ax.imshow(np.rot90(patient_data["model"].midp.imageArray[:, pointVoxelList[pointIndex][1], :]))
        ax.imshow(np.rot90(patient_data["mask"].imageArray[:, pointVoxelList[pointIndex][1], :]), alpha=0.3)
        ax.scatter([pointVoxelList[pointIndex][0]],
                   [patient_data["model"].midp.imageArray.shape[2] - pointVoxelList[pointIndex][2]],
                   c=colors[pointIndex], marker="x", s=100)
        signalAx.plot(newSignal.timestamps / 1000, signalList[pointIndex], c=colors[pointIndex])

    signalAx.set_xlabel('Time (s)')
    signalAx.set_ylabel('Deformation amplitude in Z direction (mm)')
    plt.savefig(f'{saving_path}/Patient_{patient_id}_midP_image.pdf', dpi=300)
    # plt.show()
    plt.close()

    sequenceSize = newSignal.breathingSignal.shape[0]
    subSequenceSize = maxMultiProcUse
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
            deformationList = generateDeformationListFromBreathingSignalsAndModel(patient_data["model"],
                                                                                  signalList,
                                                                                  pointList,
                                                                                  signalIdxUsed=[
                                                                                      subSequencesIndexes[i],
                                                                                      subSequencesIndexes[
                                                                                          i + 1]],
                                                                                  dimensionUsed='Z',
                                                                                  outputType=np.float32)
            # image , mask, COM 3D
            deformedImgMaskAnd3DCOMList = multiProcSpawnMethods.multiProcDeform(deformationList, patient_data["model"],
                                                                                patient_data["mask"],
                                                                                ncore=maxMultiProcUse,
                                                                                GPUNumber=GPUNumber)

            slice_index = pointVoxelList[0][1]
            if i == 0:
                plt.figure()
                plt.imshow(deformedImgMaskAnd3DCOMList[0][0].imageArray[:, slice_index, :])
                #plt.imshow(deformedImgMaskAnd3DCOMList[-1][1].imageArray[:, :, 20], alpha=0.5)
                plt.savefig(f'{saving_path}/Patient_{patient_id}_Result_Deform_serie_{serie}.pdf', dpi=300)
                plt.close()

            # print('Start multi process DRRs with', len(deformationList), 'pairs of image-mask')
            projectionResults = multiProcSpawnMethods.multiProcDRRs(deformedImgMaskAnd3DCOMList, projAngle,
                                                                    projAxis, outputSize, ncore=maxMultiProcUse)
            if i == 0:
                plt.figure()
                plt.imshow(projectionResults[-1][0], cmap='Greys')
                # plt.imshow(projectionResults[-1][1], alpha=0.5)
                plt.savefig(f'{saving_path}/Patient_{patient_id}_Result_DRR_serie_{serie}.pdf', dpi=300)
                # plt.show()
                plt.close()

            # add 3D center of mass in scanner coordinates to the result lists
            for imgIndex in range(len(projectionResults)):
                projectionResults[imgIndex].append(deformedImgMaskAnd3DCOMList[imgIndex][2])

            resultList += projectionResults
            # print('ResultList lenght', len(resultList))
            pbar.update(len(projectionResults))
    serieSavingPath = f'{saving_path}/Patient_{patient_id}_{sequenceSize}_DRRMasksAndCOM_serie_{serie}'
    saveSerializedObjects(resultList, serieSavingPath)

# end = time.time()
# print('Temps d execution', end-start)
