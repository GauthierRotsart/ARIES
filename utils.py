from opentps.opentps_core.opentps.core.io.serializedObjectIO import loadDataStructure, saveSerializedObjects
from opentps.opentps_core.opentps.core.processing.imageProcessing.imageTransform3D import getVoxelIndexFromPosition, translateData, \
    rotateData
from opentps.opentps_core.opentps.core.processing.imageProcessing.syntheticDeformation import applyBaselineShift, shrinkOrgan
from opentps.opentps_core.opentps.core.processing.segmentation.segmentation3D import getBoxAroundROI
from opentps.opentps_core.opentps.core.processing.imageProcessing.resampler3D import resample
from opentps.opentps_core.opentps.core.processing.registration.registrationRigid import RegistrationRigid
from opentps.opentps_core.opentps.core.data._patient import Patient
from visualization_tools import plot_function
import matplotlib.pyplot as plt
import copy
import numpy as np


def get_patient_data(patient_path, roi_contour_name, verbose=False):
    # load the dynamic 3D model and the RTStruct
    patient = loadDataStructure(patient_path)[0]
    model = patient.getPatientDataOfType("Dynamic3DModel")[0]
    rtstruct = patient.getPatientDataOfType("RTStruct")[0]
    if verbose:
        print('Available ROIs')
        rtstruct.print_ROINames()

    # Get the ROI and mask on which we want to apply the motion signal
    contour = rtstruct.getContourByName(roi_contour_name)
    mask = contour.getBinaryMask(origin=model.midp.origin,
                                 gridSize=model.midp.gridSize,
                                 spacing=model.midp.spacing)

    # Get the 3D center of mass of this ROI
    center_of_mass_mm = contour.getCenterOfMass(origin=model.midp.origin,
                                                gridSize=model.midp.gridSize,
                                                spacing=model.midp.spacing)
    center_of_mass_voxel = getVoxelIndexFromPosition(position=center_of_mass_mm,
                                                     image3D=model.midp)
    data = {
        "model": model,
        "mask": mask,
        "rtstruct": rtstruct,
        "com_mm": center_of_mass_mm,
        "com_voxel": center_of_mass_voxel
    }
    return data


def get_cropping_box(model, mask, rtstruct, body_contour_name, roi_contour_name):
    # define the cropping box
    gtv_box = getBoxAroundROI(mask)
    roi_contour = rtstruct.getContourByName(roi_contour_name)
    body_contour = rtstruct.getContourByName(body_contour_name)
    body_mask = body_contour.getBinaryMask(origin=model.midp.origin,
                                           gridSize=model.midp.gridSize,
                                           spacing=model.midp.spacing)
    body_box = getBoxAroundROI(body_mask)

    cropping_contours_used = [roi_contour_name, body_contour_name, roi_contour_name]  # XYZ coordinates
    cropping_box = [[], [], []]
    print(cropping_contours_used)
    for i in range(3):
        if cropping_contours_used[i] == body_contour_name:
            cropping_box[i] = body_box[i]
        elif cropping_contours_used[i] == roi_contour_name:
            cropping_box[i] = gtv_box[i]
        """else:
            raise NotImplementedError"""
    return cropping_box


def transform_patient(patient, translation, rotation, baseline_shift, shrink_size, tryGPU, GPUNumber):
    try:
        import cupy
        import cupyx

        cupy.cuda.Device(GPUNumber).use()
    except:
        print('cupy not found.')

    model = patient["model"]
    mask = patient["mask"]
    mask.imageArray = mask.imageArray.astype(float)

    # Translation
    translateData(data=model,
                  translationInMM=translation,
                  outputBox='same',
                  tryGPU=tryGPU,
                  mode='nearest',
                  fillValue=-1000)
    translateData(data=mask,
                  translationInMM=translation,
                  outputBox='same',
                  tryGPU=tryGPU,
                  mode='nearest',
                  fillValue=0)
    try:
        cupy._default_memory_pool.free_all_blocks()
    except:
        print('cupy not found.')

    # Rotation
    rotateData(data=model,
               rotAnglesInDeg=rotation,
               rotCenter='imgCenter',
               outputBox='same',
               tryGPU=tryGPU,
               mode='nearest',
               fillValue=-1000)
    rotateData(data=mask,
               rotAnglesInDeg=rotation,
               rotCenter='imgCenter',
               outputBox='same',
               tryGPU=tryGPU,
               mode='nearest',
               fillValue=0)
    try:
        cupy._default_memory_pool.free_all_blocks()
    except:
        print('cupy not found.')

    # Baseline shift
    model, mask = applyBaselineShift(inputData=model,
                                     ROI=mask,
                                     shift=baseline_shift,
                                     tryGPU=tryGPU)
    try:
        cupy._default_memory_pool.free_all_blocks()
    except:
        print('cupy not found.')

    # Shrink
    mask.imageArray[mask.imageArray > 0.5] = 1
    mask.imageArray[mask.imageArray <= 0.5] = 0
    mask.imageArray = mask.imageArray.astype(bool)

    model, mask = shrinkOrgan(model=model,
                              organMask=mask,
                              shrinkSize=shrink_size,
                              tryGPU=tryGPU)
    try:
        cupy._default_memory_pool.free_all_blocks()
    except:
        print('cupy not found.')

    patient["model"] = model
    patient["mask"] = mask
    patient["com_mm"] = mask.centerOfMass
    patient["com_voxel"] = getVoxelIndexFromPosition(position=mask.centerOfMass, image3D=model.midp)


def registration(patient, patient_T1, patient_T2, plotFig=False, saveData=False):
    patient_id = patient
    data_path = f'./DATA/4D-Lung/Patient_{patient_id}'
    fixed_patient = copy.deepcopy(patient_T1)
    moving_patient = copy.deepcopy(patient_T2)
    # PERFORM REGISTRATION
    print("Start rigid registration")
    reg = RegistrationRigid(fixed=fixed_patient["model"].midp, moving=moving_patient["model"].midp)
    transform = reg.compute()
    moving_patient["model"] = transform.deformData(moving_patient["model"], outputBox='keepAll')
    moving_patient["mask"] = transform.deformData(moving_patient["mask"], outputBox='keepAll')
    moving_patient["com_mm"] = moving_patient["mask"].centerOfMass
    moving_patient["com_voxel"] = getVoxelIndexFromPosition(position=moving_patient["com_mm"],
                                                            image3D=moving_patient["model"].midp)

    resample(fixed_patient["model"],
             spacing=[1.17, 1.17, 1.17],
             inPlace=True)
    resample(fixed_patient["mask"],
             spacing=[1.17, 1.17, 1.17],
             inPlace=True)
    fixed_patient["com_mm"] = fixed_patient["mask"].centerOfMass
    fixed_patient["com_voxel"] = getVoxelIndexFromPosition(position=fixed_patient["com_mm"],
                                                           image3D=fixed_patient["model"].midp)

    resample(moving_patient["model"],
             spacing=fixed_patient["model"].midp.spacing,
             origin=fixed_patient["model"].midp.origin,
             gridSize=fixed_patient["model"].midp.gridSize,
             fillValue=-1000,
             inPlace=True)

    resample(moving_patient["mask"],
             spacing=fixed_patient["model"].midp.spacing,
             origin=fixed_patient["model"].midp.origin,
             gridSize=fixed_patient["model"].midp.gridSize,
             fillValue=-1000,
             inPlace=True)
    moving_patient["com_mm"] = moving_patient["mask"].centerOfMass
    moving_patient["com_voxel"] = getVoxelIndexFromPosition(position=moving_patient["com_mm"],
                                                            image3D=moving_patient["model"].midp)

    border_fixed = fixed_patient["mask"].getBinaryContourMask(internalBorder=True).imageArray
    border_registered = moving_patient["mask"].getBinaryContourMask(internalBorder=True).imageArray

    diff_com = np.round(moving_patient["com_mm"] - fixed_patient["com_mm"], decimals=2).tolist()
    volume_fixed = round(fixed_patient["mask"].getVolume() / 1000, 2)
    volume_registered = round(moving_patient["mask"].getVolume() / 1000, 2)
    if plotFig:
        for axis in ['YZ', 'XZ', 'XY']:
            fig, ax = plt.subplots(2, 2, figsize=(15, 8))
            if axis == 'YZ':
                fig.suptitle(f'Patient_{patient_id} in YZ axis: \n'
                             f'Diff COM={diff_com} mm \n'
                             f'V1={volume_fixed} mL, V2={volume_registered} mL')
            elif axis == 'XZ':
                fig.suptitle(f'Patient_{patient_id} in XZ axis: \n'
                             f'Diff COM={diff_com} mm \n'
                             f'V1={volume_fixed} mL, V2={volume_registered} mL')
            elif axis == 'XY':
                fig.suptitle(f'Patient_{patient_id} in XY axis: \n'
                             f'Diff COM={diff_com} mm \n'
                             f'V1={volume_fixed} mL, V2={volume_registered} mL')
            else:
                raise ValueError

            plot_function(fig=fig,
                          ax=ax,
                          fixed_model=fixed_patient["model"],
                          fixed_mask=fixed_patient["mask"],
                          fixed_border=border_fixed,
                          fixed_com=fixed_patient["com_voxel"],
                          registered_model=moving_patient["model"],
                          registered_mask=moving_patient["mask"],
                          registered_border=border_registered,
                          registered_com=moving_patient["com_voxel"],
                          axis=axis)
        plt.show()
    if saveData:
        # Create a patient and give it the patient names
        patientName = f'Patient_{patient_id}_registered'
        patient = Patient()
        patient.name = patientName
        # Add the model and rtStruct to the patient
        patient.appendPatientData(moving_patient["model"])
        # movingModel_copy.maskList = deformedMask
        saveSerializedObjects(patient, f'{data_path}/T2/Patient_{patient_id}_registered')

    return moving_patient