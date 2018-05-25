import pyaudio, time
import wave
from datetime import datetime
import audioop

# convert rms to decibel scale decibel = 20 * log10(rms)
# http://www.sengpielaudio.com/calculator-db-volt.htm
THRESHOLD = 100 #400-600 micro Pa (40-50 dB) -> Se mide en dB -> energy threshold -> 100-317 rms (40-50 db)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
LISTEN_TIME = 5 # listening time in seconds = record time

def passiveListen():

    filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".wav"
    WAVE_OUTPUT_FILENAME = "/home/marta/Microphone/" + filename

    # starts passive listening for disturbances

    input = stream.read(CHUNK)
    rms_value = audioop.rms(input, 2)
    print rms_value

    if (rms_value < THRESHOLD):
        print "Listening...\n"

    if (rms_value >= THRESHOLD):
        print "Recording*..."

        # stores the audio data
        all = []
        # append all the chunks
        all.append(input)
        for i in range(0, RATE / CHUNK * LISTEN_TIME):
            data = stream.read(CHUNK)
            all.append(data)
        print(len(all))
        print "Recording done\n"


        # save the audio data
        data = ''.join(all)
        # stream.stop_stream()
        # stream.close()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()

    return True

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

while(True):
    passiveListen()

# import pyaudio
# pa = pyaudio.PyAudio()
#
# print '\n'.join([y['name'] for y in [pa.get_device_info_by_index(x) for x in range(pa.get_device_count())]])
#
# import pyaudio
# p = pyaudio.PyAudio()
# info = p.get_host_api_info_by_index(0)
# numdevices = info.get('deviceCount')
# for i in range(0, numdevices):
#         if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#             print "Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name')