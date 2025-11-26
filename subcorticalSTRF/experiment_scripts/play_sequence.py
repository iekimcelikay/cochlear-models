from psychtoolbox import PsychPortAudio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))  # Adjust the path as needed

from psychopy import prefs
from psychopy import hardware
from psychtoolbox import PsychPortAudio
from psychopy import sound, core
from subcorticalSTRF.stim_generator.soundgen import SoundGen


def find_audio_device(devname):
    devices = PsychPortAudio('GetDevices')
    for i in range(len(devices)):
        if devices[i]['DeviceName'] == devname:
            deviceid = devices[i]['DeviceIndex']
    return deviceid

def start_audio_device(deviceid, fs = None, channels=2, latencyclass=3):
    pahandle = PsychPortAudio('Open', deviceid, [], latencyclass, fs, channels)
    return pahandle

def play_sound(pahandle, sound_array):
    PsychPortAudio('FillBuffer', pahandle, sound_array)
    start_time = PsychPortAudio('Start', pahandle, 1)
    return start_time

def stop_audio_device(pahandle):
    status = PsychPortAudio('GetStatus', pahandle) 
    stop_result = PsychPortAudio('Stop', pahandle, 1)  # Wait for end of playback
    # Stop returns: (startTime, endPositionSecs, xruns, estStopTime)
    return {'status': status, 'stop_result': stop_result}


devname = 'HD-Audio Generic: ALC257 Analog (hw:1,0)' # if linux speaker
deviceid = find_audio_device(devname)
pahandle = start_audio_device(deviceid, fs=44100, channels=2)
sound_maker = SoundGen(sample_rate=44100, tau=0.005)# max_amplitude = 1.27 # TODO: calculate this number through the for loop simulation and it should be the same for every sound.
        # sound = sound / (max_amplitude + 0.01)  # Scale down only if neededrate=44100, tau=0.005)

print("Playing sequence with 1 harmonic (simple tones)...")
sequence_1h, frequencies = sound_maker.generate_sequence_gaussian_freq(
    freq_mean=440, freq_std=10, num_harmonics=1,
    tone_duration=0.3, harmonic_factor=0.8, 
    total_duration=3.0, isi=0.133, freq_min=410, 
    freq_max=470, seed=43, stereo=True
)
start_time = play_sound(pahandle, sequence_1h)
stop_info = stop_audio_device(pahandle)

import time
time.sleep(1)  # Pause between sequences

print("Playing sequence with 7 harmonics (rich tones)...")
sequence_7h, frequencies = sound_maker.generate_sequence_gaussian_freq(
    freq_mean=440, freq_std=10, num_harmonics=2,
    tone_duration=0.3, harmonic_factor=0.8, 
    total_duration=3.0, isi=0.133, freq_min=410, 
    freq_max=470, seed=43, stereo=True
)
start_time = play_sound(pahandle, sequence_7h)
stop_info = stop_audio_device(pahandle)


print("\nDone! The second sequence should sound richer/fuller than the first.")

# Example usage:
# deviceid = find_audio_device(devname)
# pahandle = start_audio_device(deviceid, fs=44100, channels=2)
# play_sound(pahandle, sound_array)
# stop_audio_device(pahandle)
# close_audio_device(pahandle)

# setup a meeting with Jasmin and Clem 
# try exception at the very beginning of the script 
# whenever something fails you run a piece of code that closes all the ports and stuff 
# initialize everyhting and you run the code within the try 

# to decide the amplitude do a for loop 
# iterate across all the integer frequencies 200 - 2000 Hz
# all the harmonic actord from 0.1 - 1.0 
# two for loops, you initialize this for each of the instance of the for loop you initiate a sound of a certain freq in 200 ms
# calculate the abs(max(sound)) and then store it sometwhere you want to calculate maximum across all the instances 
