import logging
import time
import numpy as np
import concurrent

# from timeit import repeat
from scipy.ndimage import zoom
from opentps.core.processing.imageSimulation.DRRToolBox import forwardProjection
from opentps.core.processing.imageProcessing.image2DManip import getBinaryMaskFromROIDRR, get2DMaskCenterOfMass
logger = logging.getLogger(__name__)

def multiProcDeform(deformationList, dynMod, GTVMask, ncore=None, GPUNumber=0):
    imgList = [dynMod.midp for i in range(len(deformationList))]
    maskList = [GTVMask for i in range(len(deformationList))]
    tryGPUList = [True for i in range(len(deformationList))]
    GPUNumberList = [GPUNumber for i in range(len(deformationList))]
    
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    nbrCore = multiprocessing.cpu_count() #number of logical cores of the machine
    resultDeformImageAndMask = []
    if ncore == None:
        logger.info("Number of logical cores used: %d", nbrCore)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(deformImageAndMask, imgList, maskList, deformationList, tryGPUList, GPUNumberList)
            resultDeformImageAndMask += results
        executor.shutdown(wait=False)
        
    elif ncore <= nbrCore:
        logger.info("Number of logical cores used: %d", ncore)
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as executor:
            results = executor.map(deformImageAndMask, imgList, maskList, deformationList, tryGPUList, GPUNumberList)
            resultDeformImageAndMask += results
        executor.shutdown(wait=False)
        
    else:
        logger.warning("Too many cores asked. The machine has less logical cores. It will use the number of logical cores of the machine.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=nbrCore) as executor:
            results = executor.map(deformImageAndMask, imgList, maskList, deformationList, tryGPUList, GPUNumberList)
            resultDeformImageAndMask += results
        executor.shutdown(wait=False)
        
    return resultDeformImageAndMask

## ------------------------------------------------------------------------------------
def deformImageAndMask(img, ROIMask, deformation, tryGPU=True, GPUNumber=0):
    """
    This function is specific to this example and used to :
    - deform a CTImage and an ROIMask,
    - compute the deformed mask 3D center of mass
    - create DRR's for both,
    - binarize the DRR of the ROIMask
    - compute the 2D center of mass for the ROI DRR
    """
    try:
        import cupy
        cupy.cuda.Device(GPUNumber).use()
    except:
        logger.warning('cupy not found.')
    
    startTime = time.time()
    image = deformation.deformImage(img, fillValue='closest', outputType=np.int16, tryGPU=tryGPU)
    mask = deformation.deformImage(ROIMask, fillValue='closest', tryGPU=tryGPU)
    centerOfMass3D = mask.centerOfMass
    logger.info(f'Image and mask deformed in {time.time() - startTime}')
    return [image, mask, centerOfMass3D]

## ----------------------------------------------------------------------------------------
def multiProcDRRs(dataList, projAngle, projAxis, outputSize, ncore=None):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    imgList = [dataList[i][0] for i in range(len(dataList))]
    maskList = [dataList[i][1] for i in range(len(dataList))]
    projAngleList = [projAngle for i in range(len(dataList))]
    projAxisList = [projAxis for i in range(len(dataList))]
    outputSizeList = [outputSize for i in range(len(dataList))]

    nbrCore = multiprocessing.cpu_count()  # number of logical cores of the machine
    croppedImgAndMaskDRRsPlus2DCOM = []
    if ncore == None:
        logger.info("Number of logical cores used: %d", nbrCore)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            croppedImgAndMaskDRRsPlus2DCOM = executor.map(DRRsBinarizeAndCrop, imgList, maskList, projAngleList,
                                                          projAxisList, outputSizeList)
        executor.shutdown(wait=False)

    elif ncore <= nbrCore:
        logger.info("Number of logical cores used: %d", ncore)
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as executor:
            results = executor.map(DRRsBinarizeAndCrop, imgList, maskList, projAngleList, projAxisList, outputSizeList)
            croppedImgAndMaskDRRsPlus2DCOM += results
        executor.shutdown(wait=True)

    else:
        logger.error("Too many cores asked. The machine has less logical cores.")

    # for element in dataList:
    #     croppedImgAndMaskDRRsPlus2DCOM.append(DRRsBinarizeAndCrop(element[0], element[1], projectionAngle=projAngle, projectionAxis=projAxis, outputSize=outputSize))

    return croppedImgAndMaskDRRsPlus2DCOM


## ------------------------------------------------------------------------------------
def DRRsBinarizeAndCrop(image, mask, projectionAngle=0, projectionAxis='Z', outputSize=[]):
    startTime = time.time()
    DRR = forwardProjection(image, projectionAngle, axis=projectionAxis)
    DRRMask = forwardProjection(mask, projectionAngle, axis=projectionAxis)
    rowsToRemove = []
    for i in range(DRR.shape[0]):
        if np.std(DRR[i, :]) == 0:
            rowsToRemove.append(i)

    if rowsToRemove:
        rowsToRemove.reverse
        DRR = np.delete(DRR, rowsToRemove, 0)
        DRRMask = np.delete(DRRMask, rowsToRemove, 0)

    columnsToRemove = []
    for i in range(DRR.shape[1]):
        if np.std(DRR[:, i]) == 0:
            columnsToRemove.append(i)

    if columnsToRemove:
        columnsToRemove.reverse
        DRR = np.delete(DRR, columnsToRemove, 1)
        DRRMask = np.delete(DRRMask, columnsToRemove, 1)

    # if outputSize:
    # print('Before resampling')
    # print(DRR.shape, np.min(DRR), np.max(DRR), np.mean(DRR))
    # ratio = [outputSize[0] / DRR.shape[0], outputSize[1] / DRR.shape[1]]
    # DRR = zoom(DRR, ratio)
    # DRRMask = zoom(DRRMask, ratio)
    # print('After resampling')
    # print(DRR.shape, np.min(DRR), np.max(DRR), np.mean(DRR))

    h, w = DRR.shape
    if h != w:
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        DRR = np.pad(DRR, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), mode='constant', constant_values=0)
        DRRMask = np.pad(DRRMask, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), mode='constant',
                         constant_values=0)
    ratio = [outputSize[0] / DRR.shape[0], outputSize[1] / DRR.shape[1]]
    DRR = zoom(DRR, ratio)
    DRRMask = zoom(DRRMask, ratio)
    del image  # to release the RAM
    del mask  # to release the RAM

    binaryDRRMask = getBinaryMaskFromROIDRR(DRRMask)
    centerOfMass = get2DMaskCenterOfMass(binaryDRRMask)
    # return [DRR, DRRMask]
    logger.info(f'DRR projection done in {time.time() - startTime}')
    return [DRR, binaryDRRMask, centerOfMass]

## ------------------------------------------------------------------------------------