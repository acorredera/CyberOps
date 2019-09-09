import os, shutil,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import matlab.engine
import pyaudio
import wave
import UTILS.dataLoader as loader
import numpy as np
import AUDIO.normalization as normalization
import AUDIO.FFT_image_generator as FFTimgGen
import AUDIO.FFTImageDataGenerator as FFTImageDataGenerator
import AUDIO.OpenSMILE_extractor.functionals_extractor as funct_extractor
import datetime
from kafka import KafkaProducer
import json
import settings
import audioop, math


#USAR MATLAB DESDE PYTHON
#https://es.mathworks.com/help/matlab/matlab_external/call-user-script-and-function-from-python.html
#https://es.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
# En sistemas Windows —
# cd "matlabroot\extern\engines\python"
# python setup.py install
# En Mac o sistemas Linux —
# cd "matlabroot/extern/engines/python"
# python setup.py install


def record_audio_continuous(audio, FORMAT, CHANNELS,RATE, CHUNK):
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    #print("recording...")
    return stream

def save_audio(audio,stream,FORMAT, CHANNELS,RATE, CHUNK,RECORD_SECONDS, WAVE_OUTPUT_FILENAME):

    frames = []
    frames_noise = []
    if(os.path.isfile(WAVE_OUTPUT_FILENAME)):
        os.remove(WAVE_OUTPUT_FILENAME)
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        #if(i % int(RATE / CHUNK * 5) == 0):
            # OBTAINING NOISE VALUE:


    #print("finished recording")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    return frames



if __name__ == "__main__":
    producer = KafkaProducer(bootstrap_servers=settings.ip_kafka_DCOS, api_version=(0, 10, 1))
    topic_emotion = "cyberops_arousal"

    producer_noise = KafkaProducer(bootstrap_servers=settings.ip_kafka_DCOS, api_version=(0, 10, 1)) #localhost:9092
    #OPTION 1: FUNCTIONALS -- IN PROCESS OF IMPLEMENTATION
    #PATH_OPENSMILE = "/home/cris/PycharmProjects/CYBEROPS/"
    #input_audio_path = "/media/cris/MyPassport/a-INFO/BECAS/BECA-DIE/DATASETS/DS_audio/RECOLA_dataset/avec_2016/recordings_audio/dev_1.wav"
    #output_arff_with_funcionals = "/media/cris/MyPassport/a-INFO/BECAS/BECA-DIE/DATASETS/DS_audio/RECOLA_dataset/avec_2016/recordings_audio/dev_1_mid_v2.arff"
    #config_file_path = os.path.join(PATH_OPENSMILE, "AAdata/libs/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf")
    # funct_extractor.get_OpenSmile_functionals(PATH_OPENSMILE, input_audio_path, config_file_path,
    #                                           output_arff_with_funcionals, include_class=False, classes_dict={0},
    #                                           current_class=0, exportationType=".arff")

    # print("Spectrum generated!!")
    # #LOAD ARFF AND REPLICATE DATA
    # arff_data_array, arff_header, _, _ = loader.loadCompleteArff(output_arff_with_funcionals, columns2delete=[-1], stringAttr = False)
    # first_row = arff_data_array[50, :]
    # last_row = arff_data_array[-1,:]
    # new_final = np.ones(shape=(51,arff_data_array.shape[-1]))*last_row
    # arff_data_array[0:50,:] = np.ones(shape=(50,arff_data_array.shape[-1]))*first_row
    # arff_data_array = np.concatenate((arff_data_array, new_final))
    # print("arff with functionals generated")
    PATH_ini = os.getcwd()#
    PATH = os.path.join(PATH_ini, "AUDIO") #
    PATH_mirtoolbox = os.path.join(PATH_ini, "AAdata/libs/MIRtoolbox1.7")
    print(PATH_mirtoolbox)
    #OPTION 2: SPECTRUM PREDICTIONS
    # #CREATE SPECTRUM (.MAT)
    #START MATLAB UNION --> IMPORTANT: PREVIOUSLY INSTALL MATLAB SETUP.PY
    eng1 = matlab.engine.start_matlab()
    eng1.addpath(PATH, nargout=0)
    eng1.addpath(PATH_mirtoolbox, nargout=0)

    # LOAD MODEL
    path_model = os.path.join(PATH, "AAdata/models/cnn_arousal/Recola_audio_CNN_FFT_lastLayers_TranfLearn_arousal_option_1_optimizer_rms_lr_0.0001_lossf_mse_batch_32_norm_7_codeChannels_3_EpochFinalModel_10")
    model = loader.loadModelKeras(weights_path=path_model + ".h5", json_path=path_model + ".json")
    #Parameters:
    root_folder = os.path.join(PATH,"AAdata/auto_generated_data/")
    # - spectrum generation parameters
    fs = 16000.0
    N = 1343.0
    fsize = 0.1
    fstep = 0.04
    output_folder_spectrum = os.path.join(root_folder, "audio_spectrums/")
    # - normalization parameters
    typeOfNormalization = 7
    newSupValue = 255
    newInfValue = 0
    text2add = "norm"
    th = None
    percentileNormalizationType4 = 25
    codeChannels = 3
    # - cutting image parameters
    sizeSquareImages = 224
    segments_per_image = 2 #higher number, higher overlappings
    output_imgs_folder = os.path.join(root_folder, "audio_images/")
    output_img_folder_final = os.path.join(root_folder, "audio_images_final/")

    #AUDIO RECORDING PARAMETERS:
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 20# min 10-15 seconds for obtaining correct images
    aux_audio_folder = os.path.join(PATH, "AAdata/auto_generated_data/")
    #NOISE PARAMETERS:
    THRESHOLD1 = 3162  # 100-317 rms (40-50 db); 70 dB = 3162 rms
    THRESHOLD2 = 10000  # 80 dB = 10000 rms


    #Create folders:
    if(not(os.path.exists(aux_audio_folder))):
        os.makedirs(aux_audio_folder+"audio/")
        os.makedirs(aux_audio_folder + "audio_functionals/")
        os.makedirs(aux_audio_folder + "audio_images/")
        os.makedirs(aux_audio_folder + "audio_images_final/")
        os.makedirs(aux_audio_folder + "audio_spectrums/")
    else:
        shutil.rmtree(aux_audio_folder)
        os.makedirs(aux_audio_folder + "audio/")
        os.makedirs(aux_audio_folder + "audio_functionals/")
        os.makedirs(aux_audio_folder + "audio_images/")
        os.makedirs(aux_audio_folder + "audio_images_final/")
        os.makedirs(aux_audio_folder + "audio_spectrums/")

    WAVE_OUTPUT_FILENAME =os.path.join(PATH, "AAdata/auto_generated_data/audio/")
    #FUNCIONALS/PATHS OF AUXILIAR VARIABLES PARAMETERS
    print("Start recording ...")
    audio = pyaudio.PyAudio()
    stream = record_audio_continuous(audio, FORMAT, CHANNELS, RATE, CHUNK)

    #config_file_path = os.path.join(PATH_OPENSMILE, "AAdata/libs/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf")
    #functionals_output_arff = os.path.join(PATH_OPENSMILE, "AAdata/auto_generated_data/audio_functionals/aux_audio")

    while(True):
        for aux_index in range(0,4):
            #aux_index+=1
            #create audio
            tm = datetime.datetime.now()
            input_audio_path = WAVE_OUTPUT_FILENAME+"aux_audio"+str(aux_index)+".wav"
            frames = save_audio(audio,stream,FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, input_audio_path)
            #create funcional
            #output_arff_with_funcionals = functionals_output_arff+str(aux_index)+".arff"
            #funct_extractor.get_OpenSmile_functionals(PATH_OPENSMILE,input_audio_path, config_file_path, output_arff_with_funcionals, include_class=False, classes_dict = {0},current_class=0, exportationType=".arff")
            # CREATE SPECTURM & PREDICTION:
            dataId = "aux_audio"+str(aux_index)+ ".mat"
            output_path_spectrum_out = os.path.join(output_folder_spectrum, dataId)

            eng1.python_FFT_spectrum(input_audio_path, output_path_spectrum_out, fs, N, fsize, fstep)
            # NORMALIZATION OF SPECTRUM
            normalization.dataNormalization(dataId=dataId, inputDataFolder=output_folder_spectrum,
                                            outputPath=output_folder_spectrum,
                                            typeOfNormalization=typeOfNormalization,
                                            newSupValue=newSupValue,
                                            newInfValue=newInfValue,
                                            text2add=text2add,
                                            zscoreThreshold=th,
                                            percentileNormalizationType4=percentileNormalizationType4)

            # #CREATE IMAGES
            input_mat_path = output_folder_spectrum
            mat_file_name = dataId.split(".mat")[0] + text2add + ".mat"
            FFTimgGen.create_imgs_from_spectrums_parallel(mat_file_name, input_mat_path, output_imgs_folder,
                                                          sizeSquareImages, codeChannels=codeChannels)

            # #CUT IMAGES
            image_name = mat_file_name.split(".mat")[0]
            # OPTION 1: CUT IMAGES WITHOUTH OVERLAPPING
            # FFTimgGen.cutSpectrumImage(image_name,
            # inputImagesFolder = output_imgs_folder,
            # outputSpectrumSlicesFolder = output_img_folder_final,
            # sizeSquareImages = sizeSquareImages,
            # additionalText = ".png")
            # OPTION 2: CUT IMAGES WITH OVERLAPPING
            FFTimgGen.cutSpectrumImage_dataAugm(audioRecordingId=image_name,
                                                inputImagesFolder=output_imgs_folder,
                                                outputSpectrumSlicesFolder=output_img_folder_final,
                                                sizeSquareImages=sizeSquareImages,
                                                increment_negatives=segments_per_image,
                                                additionalText=".png")
            # PREDICTIONS
            list_imgs_2_predict = sorted(os.listdir(output_img_folder_final))
            dataIdInfo = dataId.split(".mat")[0]
            new_list = []
            for img in list_imgs_2_predict:
                if(dataIdInfo in img):
                    new_list+=[img]
            labels = dict.fromkeys(new_list, 0)
            batch_size = len(new_list)
            test_generatorPredictions = FFTImageDataGenerator.FFTImageDataGenerator(new_list, labels=labels,
                                                                                    batch_size=batch_size,
                                                                                    imagesFolder=output_img_folder_final,
                                                                                    shuffle=False)
            yhatTest = model.predict_generator(generator=test_generatorPredictions,
                                               steps=len(test_generatorPredictions),
                                               use_multiprocessing=True)
            pred = yhatTest.reshape(-1)
            final_pred = np.mean(pred)

            #SEND DATA TO CASSANDRA - emotion
            print("Score for audio (" + dataId + "): " + str(final_pred))
            timestamp_rounded = tm - datetime.timedelta(seconds=tm.second, microseconds=tm.microsecond)
            data2send = {"timestamp": timestamp_rounded.strftime("%Y-%m-%d %H:%M:%S"),  # redondeado
                         "arousal": str(final_pred),
                         "employeeAlias":settings.employee_name}
            producer.send(topic_emotion, json.dumps(data2send).encode('utf-8'))
            # SEND DATA TO CASSANDRA - noise
            rms_value = audioop.rms(frames[0], 2)
            db = int(round(20 * math.log(rms_value, 10)))
            print(db)
            # send noise data
            data2send_noise = {"timestamp": timestamp_rounded.strftime("%Y-%m-%d %H:%M:%S"),  # redondeado
                         "db": db,
                         "room_id": "1"}
            producer_noise.send("cyberops_noise_level", json.dumps(data2send_noise).encode('utf-8'))



            #SEND DATA TO CASSANDRA...
        #REMOVE DATA
        #os.remove(input_audio_path)
        print("removing files auto-generated...")
        shutil.rmtree(aux_audio_folder + "audio_functionals/")
        shutil.rmtree(aux_audio_folder + "audio_images/")
        shutil.rmtree(aux_audio_folder + "audio_images_final/")
        shutil.rmtree(aux_audio_folder + "audio_spectrums/")
        #create folders again
        os.makedirs(aux_audio_folder + "audio_functionals/")
        os.makedirs(aux_audio_folder + "audio_images/")
        os.makedirs(aux_audio_folder + "audio_images_final/")
        os.makedirs(aux_audio_folder + "audio_spectrums/")


