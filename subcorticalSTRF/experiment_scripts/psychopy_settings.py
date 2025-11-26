from pprint import pprint
import psychtoolbox.audio
pprint(psychtoolbox.audio.get_devices())

from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']

from psychtoolbox import audio

# Get all audio devices
devices = audio.get_devices()

for idx, dev in enumerate(devices):
    print(f"\nDevice {idx}:")
    print(f"  Name: {dev.get('DeviceName', 'N/A')}")
    print(f"  Output Channels: {dev.get('NrOutputChannels', 0)}")
    print(f"  Input Channels: {dev.get('NrInputChannels', 0)}")
    print(f"  Device Index: {dev.get('DeviceIndex', 'N/A')}")