from psychtoolbox import PsychPortAudio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
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

# Setup audio
devname = 'HD-Audio Generic: ALC257 Analog (hw:1,0)'
deviceid = find_audio_device(devname)
pahandle = start_audio_device(deviceid, fs=44100, channels=2)
sound_maker = SoundGen(sample_rate=44100, tau=0.005)

print("Playing 1 harmonic (pure tone)...")
tone_1 = sound_maker.sound_maker(freq=440, num_harmonics=1, tone_duration=1.5, harmonic_factor=0.7)
# Normalize to prevent clipping
tone_1 = tone_1 / np.max(np.abs(tone_1)) * 0.5
tone_1_stereo = np.column_stack((tone_1, tone_1))
play_sound(pahandle, tone_1_stereo)
stop_audio_device(pahandle)

import time
time.sleep(0.5)  # Pause between sounds

print("Playing 7 harmonics (rich/complex tone)...")
tone_7 = sound_maker.sound_maker(freq=440, num_harmonics=7, tone_duration=1.5, harmonic_factor=0.7)
# Normalize to prevent clipping
tone_7 = tone_7 / np.max(np.abs(tone_7)) * 0.5
tone_7_stereo = np.column_stack((tone_7, tone_7))
play_sound(pahandle, tone_7_stereo)
stop_audio_device(pahandle)

close_audio_device(pahandle)
print("\nDone! The first sound should be a pure sine wave, the second should sound richer/fuller.")
