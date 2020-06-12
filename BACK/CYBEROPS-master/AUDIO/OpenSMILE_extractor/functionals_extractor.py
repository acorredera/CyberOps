import subprocess,os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


#exportationType: csv, arff
def get_OpenSmile_functionals(PATH_OPENSMILE, csv_audio_path, config_file_path, result_file_path, include_class=False, classes_dict = {0},current_class=0, exportationType=".arff"):
    openSmile_path = os.path.join(PATH_OPENSMILE, 'AAdata/libs/opensmile-2.3.0/')+"SMILExtract -C "
    configuration = config_file_path
    input= " -I "
    audio = csv_audio_path
    if(exportationType == ".arff"):
        output = " -O "
    elif(exportationType == ".csv"):
        output = " -csvoutput "
    else:
        output = " -O "
    # SMILExtract - C
    # config / emobase_live4_batch.conf - I
    # example - audio / opensmile.wav > result.txt
    command = "".join([openSmile_path,configuration, input, audio, output, result_file_path])

    if(include_class):
        command = "".join([command, " -classtype ",str(classes_dict).replace(" ",""), " -class ",str(current_class)])
    print(command)
    # command = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn %s"%video_path, video_id
    a = subprocess.call(command, shell=True)
    print(a)


