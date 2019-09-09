# CYBEROPS
Para hacer request de imágenes, una vez pongamos el geeterNoticiasServlet a correr enviamos la petición como en el ejemplo:
http://127.0.0.1:5002/static/images/perro.jpg

# GUÍA RÁPIDA INSTALACIÓN

GUÍA INSTALACIÓN “RECONOCIMIENTO FACIAL DE EMOCIONES”

1.- Fisher Face Recognicer → http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/

	1.- OpenCv → CV2

“sudo apt-get install libopencv-dev python-opencv”
http://milq.github.io/install-opencv-ubuntu-debian/

“pip install opencv-python”
https://stackoverflow.com/questions/21792909/cv2-videocapture-open-always-returns-false


2.- Lands Marks Classifier → http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/

	1.- OpenCv

	2.- Sklearn → “pip install sklearn”

	3.- Dlib
		3.1.- Install Cmake → https://cmake.org/download/
		3.2.- Download Boost-Python and extract the package → http://www.boost.org/

	Download boost_1_66_0.tar.bz2.
1. In the directory where you want to put the Boost installation, execute
tar --bzip2 -xf /path/to/boost_1_66_0.tar.bz2
http://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html
2. Run either .\bootstrap.bat (on Windows), or ./bootstrap.sh (on other operating systems).
Run “./b2 install”
http://www.boost.org/doc/libs/1_55_0/doc/html/bbv2/installation.html
https://codeyarns.com/2017/01/24/how-to-build-boost-on-linux/
1. 3. Face Landmark Detection → http://dlib.net/face_landmark_detection.py.html
2. 4. https://gist.github.com/talegari/f835763b26c94a5790d5f2b298a64c33 →
For ubuntu 16.04:
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
Assuming pip is already installed:
sudo apt-get install libboost-dev
For python bindings:
sudo apt-get install libboost-python-dev
sudo pip install numpy
sudo pip install scipy
sudo pip install scikit-image

sudo pip install dlib
https://gist.github.com/talegari/f835763b26c94a5790d5f2b298a64c33

GUÍA INSTALACIÓN PYAUDIO
you download PyAudio-0.2.11.tar.gz (md5) in https://pypi.python.org/pypi/PyAudio#downloads and extract file.
cd PyAudio-0.2.11
sudo python setup.py install
https://stackoverflow.com/questions/44538746/cannot-install-pyaudio-0-2-11-in-ubuntu-16-04


OTRAS PÁGINAS DE INTERÉS RECONOCIMIENTO FACIAL EMOCIONES WEBCAM

https://github.com/mihaelacr/pydeeplearn/tree/master/code/webcam-emotion-recognition

https://github.com/sarxos/webcam-capture/tree/master/webcam-capture-examples/webcam-capture-motiondetector

https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_api.html


OTRAS PÁGINAS DE INTERÉS PYAUDIO

https://github.com/tyiannak/pyAudioAnalysis

https://github.com/tyiannak/pyAudioAnalysis/issues/1



#RUNNING PROGRAM
1st time:
1. Create tables in cassandra(function create_table() in __DAOImpl) and kafka topics:

Create kafka topics in the DCOS:
(-r → replication factor (for redundancy, always < number of kafka brokers), -p → partitions (for paralellisation
    • dcos kafka --name=kafka-services/kafka topic create cyberops_HR -r 2 -p 1
    • dcos kafka --name=kafka-services/kafka topic create cyberops_temperature -r 2 -p 1
    • dcos kafka --name=kafka-services/kafka topic create cyberops_noise_level -r 2 -p 1
    • dcos kafka --name=kafka-services/kafka topic create cyberops_emotion -r 2 -p 1
    • dcos kafka --name=kafka-services/kafka topic create cyberops_alexaAnswer -r 2 -p 1
      dcos kafka --name=kafka-services/kafka topic create cyberops_arousal -r 2 -p 1
      dcos kafka --name=kafka-services/kafka topic create cyberops_machines -r 2 -p 1


2. Adjust settings.py with the data of the server, operator…

3. Launch iniziatice_server.sh in the server or computer acting as server & initialize_operator_recordings.sh in the operator computer.

      bash  iniziatice_server.sh
      bash  initialize_operator_recordings.sh
      bash  initialize_machines.sh

4. Re-start humidity/temperature sensor (press black button in sensor)

5. Re-start pebble app (open the app ‘Real_Time_HR_post’ in the peeble)


** to stop .sh scripts → killall python3

#CASSANDRA

Delete the whole data of a specific table: use the command TRUNCATE ("TRUNCATE cyberops.employeestatedb;")


#MATLAB

HOW TO USE MATLAB SCRIPT IN PYTHON PROGRAM
https://es.mathworks.com/help/matlab/matlab_external/call-user-script-and-function-from-python.html
https://es.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
 En sistemas Windows —
 cd "matlabroot\extern\engines\python"
 python setup.py install
 En Mac o sistemas Linux —
 cd "matlabroot/extern/engines/python"
 python setup.py install



#Fill requirements.txt automatically:
pip freeze > requirements.txt   #in the current working directory