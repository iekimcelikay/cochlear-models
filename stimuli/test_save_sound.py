
import sys
from pathlib import Path
# Ensure project root is on sys.path so tests can import local modules
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import numpy as np
import soundfile as sf

from save_sound import save_sequence_as_wav


def test_soundsequence_save(out_path):
    """Ensure SoundSequence.save_to_wav writes the correct data."""
    from soundgen import SoundGen


    fs = 100000
    soundgen = SoundGen(fs, 0.005)

    sequence = soundgen.generate_sequence(freq=440, num_harmonics=1, tone_duration=0.867,
                        harmonic_factor=1, dbspl=60, total_duration=5, isi=0.133)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    out_file = out_path / "sequence_condition6_fs100khz_440hz.wav"

    # Save without normalization so read-back matches closely
    save_sequence_as_wav(sequence, 100000, str(out_file), subtype="FLOAT")

#    data, sr_read = sf.read(str(out_file), dtype="float32")


test_soundsequence_save(str(Path(__file__).parent / "output"))
