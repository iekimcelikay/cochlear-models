from psychtoolbox import PsychPortAudio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

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
    stop_result = PsychPortAudio('Stop', pahandle, 1)
    return stop_result

def close_audio_device(pahandle):
    PsychPortAudio('Close', pahandle)

# Setup
devname = 'HD-Audio Generic: ALC257 Analog (hw:1,0)'
deviceid = find_audio_device(devname)
pahandle = start_audio_device(deviceid, fs=44100, channels=2)
sound_maker = SoundGen(sample_rate=44100, tau=0.005)

print("Playing sequence with 1 harmonic (simple tones)...")
# Use CONSTANT frequency (no Gaussian variation) to hear harmonic difference
sequence_1h = sound_maker.generate_sequence(
    freq=440, 
    num_harmonics=1,
    tone_duration=0.2, 
    harmonic_factor=0.7, 
    total_duration=3.0, 
    isi=0.133, 
    stereo=True
)
play_sound(pahandle, sequence_1h)
stop_audio_device(pahandle)

import time
time.sleep(1)

print("Playing sequence with 7 harmonics (rich tones)...")
sequence_7h = sound_maker.generate_sequence(
    freq=440, 
    num_harmonics=7,
    tone_duration=0.2, 
    harmonic_factor=0.7, 
    total_duration=3.0, 
    isi=0.133, 
    stereo=True
)
play_sound(pahandle, sequence_7h)
stop_audio_device(pahandle)

close_audio_device(pahandle)
print("\nDone! The second sequence should sound much richer/fuller than the first.")
