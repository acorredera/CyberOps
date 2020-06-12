import os, math, shutil
import numpy as np
from PIL import Image
import UTILS.dataLoader as loader
import UTILS.dataSaver as saver
import AUDIO.normalization as normalization
import matplotlib as matplot
matplot.use('Agg')
import matplotlib.pyplot as mpl


def cutSpectrumMat(audioRecordingId,inputSpectrumFolder,outputSpectrumSlicesFolder, sizeSquareImages, additionalText = '_spectrum.mat', structName='new_spectrum'):
    spectrumFilePath = "".join(
        [inputSpectrumFolder, audioRecordingId, additionalText])
    if (os.path.exists(spectrumFilePath)):
        spectrum_data = (loader.loadMatFiles(spectrumFilePath))[structName]
        sizeSpectrum = np.shape(spectrum_data)
        rst = sizeSpectrum[-1] % sizeSquareImages
        if (rst != 0):
            cols2add = sizeSquareImages - rst
            minimum = spectrum_data.min()
            spectrum_data = np.hstack([np.ones([sizeSpectrum[0], cols2add]) * minimum, spectrum_data])
        numSlices = int((np.shape(spectrum_data)[1]) / sizeSquareImages)
        print("CREATING SLICES: nImages", numSlices, " for file: ", audioRecordingId)
        for sliceIndex in range(0, numSlices):
            indexColumnStart = (sliceIndex) * sizeSquareImages
            indexColumnEnd = ((sliceIndex + 1) * sizeSquareImages)
            spectrum_data_slice = spectrum_data[0:sizeSquareImages, indexColumnStart:indexColumnEnd]
            imgOutputPath = "".join(
                [outputSpectrumSlicesFolder, "/", audioRecordingId, "_", str(sliceIndex),'.mat'])
            saver.saveMatrixAsMat(spectrum_data_slice,imgOutputPath, nameMatrixMatlab=structName)

def cutSpectrumImage(audioRecordingId,inputImagesFolder,outputSpectrumSlicesFolder, sizeSquareImages, additionalText=".png"):
    spectrumFilePath_image = "".join([inputImagesFolder, audioRecordingId, additionalText])
    if (os.path.exists(spectrumFilePath_image)):
        audioRecordingId = audioRecordingId.replace(".png","")
        spectrum_data = loader.loadImageAsArray(spectrumFilePath_image)
        sizeSpectrum = np.shape(spectrum_data)
        if(len(sizeSpectrum)>=3):
            rst = sizeSpectrum[1] % sizeSquareImages
            if (rst != 0):
                cols2add = sizeSquareImages - rst
                spectrum_data = np.hstack([np.ones([sizeSpectrum[0], cols2add,3]) * 0, spectrum_data])
            numSlices = int((np.shape(spectrum_data)[1]) / sizeSquareImages)
            print("CREATING SLICES: nImages", numSlices, " for file: ", audioRecordingId)
            for sliceIndex in range(0, numSlices):
                indexColumnStart = (sliceIndex) * sizeSquareImages
                indexColumnEnd = ((sliceIndex + 1) * sizeSquareImages)
                spectrum_data_slice = spectrum_data[0:sizeSquareImages, indexColumnStart:indexColumnEnd]
                imgOutputPath = "".join(
                    [outputSpectrumSlicesFolder, "/", audioRecordingId, "_", str(sliceIndex), '.png'])
                try:
                    img = Image.fromarray(spectrum_data_slice.astype('uint8'), 'RGB')
                    saver.save_img(img, imgOutputPath)
                except:
                    print("----------------------Error en imagen: ", audioRecordingId)
        else:
            print("NOT IMPLEMENTED YET!!, Use 3channel Images")
            rst = sizeSpectrum[-1] % sizeSquareImages
            if (rst != 0):
                cols2add = sizeSquareImages - rst
                spectrum_data = np.hstack([np.ones([sizeSpectrum[0], cols2add]) * 0, spectrum_data])

def cutSpectrumImage_dataAugm(audioRecordingId,inputImagesFolder,outputSpectrumSlicesFolder, sizeSquareImages,increment_negatives=7, additionalText=".png"):
    spectrumFilePath_image = "".join([inputImagesFolder, audioRecordingId, additionalText])
    if (os.path.exists(spectrumFilePath_image)):
        audioRecordingId = audioRecordingId.replace(".png","")
        spectrum_data = loader.loadImageAsArray(spectrumFilePath_image)
        sizeSpectrum = np.shape(spectrum_data)
        if(len(sizeSpectrum)>=3):
        #diferenciar entre negativo/positivo
            numSlices = int((np.shape(spectrum_data)[1]) / sizeSquareImages)
    w_ini = sizeSpectrum[1]
    augmented_num_slices = increment_negatives * numSlices  # Generate 7 times more data for non-negatives samples by overlapping more data
    non_overlapping = int((w_ini - sizeSquareImages) / augmented_num_slices)
    overlapping = sizeSquareImages - non_overlapping
    print("NEGATIVE FILE -- CREATING SLICES: nImages", augmented_num_slices, " for file: ", audioRecordingId)
    indexColumnStart = 0
    for sliceIndex in range(0, augmented_num_slices):
        indexColumnEnd = indexColumnStart + sizeSquareImages
        spectrum_data_slice = spectrum_data[0:sizeSquareImages, indexColumnStart:indexColumnEnd]
        indexColumnStart = indexColumnStart + non_overlapping
        imgOutputPath = "".join(
            [outputSpectrumSlicesFolder, "/", audioRecordingId, "_", str(sliceIndex), '.png'])
        try:
            img = Image.fromarray(spectrum_data_slice.astype('uint8'), 'RGB')
            saver.save_img(img, imgOutputPath)
        except:
            print("----------------------Error en imagen: ", audioRecordingId)
            # if(sliceIndex==augmented_num_slices-2):
            #     print("debug")


def create_imgs_from_spectrums_parallel(mat_file_name,input_mat_path, output_imgs_folder, sizeSquareImages, codeChannels=32):
    # cration of images with(out) labels
    input_mat_path = "".join([input_mat_path,mat_file_name])
    fileName = mat_file_name.split(".mat")[0]
    print("Processing image: "+mat_file_name)
    if (os.path.exists(input_mat_path) and ".mat" in input_mat_path):
        spectrum_data = (loader.loadMatFiles(input_mat_path))['fixed_spectrum_matrix']
        out_path = "".join([output_imgs_folder, fileName, ".png"])
        img, _, _ = getImageByCodeChannel(codeChannels,spectrum_data, sizeSquareImages, mat_file_name)
        saver.save_img(img, out_path)


def create_imgs_from_spectrums(mat_folder, output_imgs_folder, codeChannels=32):
    # cration of images with(out) labels
    listSpectrumFiles = sorted(os.listdir(mat_folder))
    for spectrum_file in listSpectrumFiles:
        fileName = spectrum_file.split(".")
        if (".mat" in spectrum_file):
            print("Processing image: "+spectrum_file)
            input_file_path = "".join([mat_folder, spectrum_file])
            spectrum_data = (loader.loadMatFiles(input_file_path))['fixed_spectrum_matrix']
            sizeSpectrum = np.shape(spectrum_data)
            out_path = "".join([output_imgs_folder, fileName[0], ".png"])
            img, _, _ = getImageByCodeChannel(codeChannels,spectrum_data, sizeSpectrum[0], spectrum_file)
            saver.save_img(img, out_path)




def getImageByCodeChannel(codeChannels,spectrum_data,nrows, nameImg=""):
    spectrum_data_shape = np.shape(spectrum_data)
    if (codeChannels == 3):
        imgColor = "BGR4SqNet"
        new_image = np.ones([nrows, spectrum_data_shape[1], 3], dtype=np.uint8)
        new_image[:, :, 0] = np.rint(spectrum_data[0:nrows, 0::])
        new_image[:, :, 1] = np.rint(spectrum_data[nrows: nrows * 2, 0::])
        new_image[:, :, 2] = np.rint(spectrum_data[((nrows * 2))::, 0::])
        img = Image.fromarray(new_image.astype('uint8'), 'RGB')
    elif (codeChannels == 31):  # the difference is that we introduce the same image in the 3 channels
        imgColor = "GREY-31"  # we maintain the fisr frequencies -- check this part--
        new_image = np.ones([nrows, spectrum_data_shape[1], 3], dtype=np.uint8)
        new_image[:, :, 0] = np.rint(spectrum_data[0:nrows, 0::] / 3)
        new_image[:, :, 1] = np.rint(spectrum_data[0:nrows, 0::] / 3)
        new_image[:, :, 2] = np.rint(spectrum_data[0:nrows, 0::] / 3)
        img = Image.fromarray(new_image.astype('uint8'), 'RGB')  # if it is grey, 'L'
    elif (codeChannels == 311):
        imgColor = "GREY-311"  # we maintain the fisr frequencies -- check this part--
        new_image = np.ones([nrows, spectrum_data_shape[1], 3], dtype=np.uint8)
        new_image[:, :, 0] = np.rint(spectrum_data[0:nrows, 0::])
        new_image[:, :, 1] = np.rint(spectrum_data[0:nrows, 0::])
        new_image[:, :, 2] = np.rint(spectrum_data[0:nrows, 0::])
        img = Image.fromarray(new_image.astype('uint8'), 'RGB')  # if it is grey, 'L'
    elif (codeChannels==1):
        imgColor = "GREY-1"  # we maintain the fisr frequencies -- check this part--
        new_image = np.ones([nrows, spectrum_data_shape[1]], dtype=np.uint8)
        new_image[:, :] = np.rint(spectrum_data[0:nrows, 0::])
        img = Image.fromarray(new_image.astype('uint8'), 'L')
    elif (codeChannels == 32): #probado
        imgColor = "BGRjet4SqNet"  # we maintain the fisr frequencies -- check this part--
        cm_jet = mpl.cm.get_cmap('jet')
        avg, std, min, max = normalization.get_statistics_from_data(spectrum_data, axis_attributes=None)
        print("Max: " + str(max) + ", min: " + str(min)+"for img: "+ nameImg)
        espectro0255to01 = normalization.getNormalizedData(spectrum_data, min, max, newSupValue=1, newInfValue=0)
        new_image2 = (cm_jet(espectro0255to01)) * 255  # img thus it has image coordinates
        new_image = new_image2[0:nrows, 0::, 2::-1]
        img = Image.fromarray(new_image.astype('uint8'), 'RGB')
        #new_image = new_image2[0:sizeSquareImages, indexColumnStart:indexColumnEnd, 2::-1]
    return img,imgColor, new_image
