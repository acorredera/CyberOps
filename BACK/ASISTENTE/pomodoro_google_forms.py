import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import settings
import webbrowser
from pydub import AudioSegment
from pydub.playback import play
import ASISTENTE.Extractor_Firebase as firebase_extractor

def forms_request(path_sound,previous_path=""):
    while(True):
        #os.system("aplay ASISTENTE/Metronome-Sound.wav")
        #browser = webbrowser.get("/usr/bin/google-chrome %s")
        url = 'https://console.actions.google.com/project/cyberops-firebase-v1/simulator'
        #chrome_path = '/usr/bin/google-chrome %s'
        webbrowser.open(url, new=1, autoraise=True)
        song = AudioSegment.from_wav("".join([previous_path, path_sound]))
        play(song) #aquí podemos grabar un vídeo y que diga google talk to cyberopss application
        time.sleep(150) #360
        firebase_extractor.remove_and_upload_firebase_data()
        print("Written AAdata into Cassandra")
        time.sleep(600) #1440
        #time.sleep(200)
        #podemos meter aquí el extractor firebase para que vaya borrando

if __name__ == '__main__':
    path_sound = os.getcwd() + "/ASISTENTE/Metronome-Sound.wav"
    forms_request(path_sound)