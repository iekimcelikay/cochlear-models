"""Unit tests for save_sound.save_sequence_as_wav"""

import sys
from pathlib import Path
# Ensure project root is on sys.path so tests can import local modules
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import numpy as np
import soundfile as sf

from save_sound import save_sequence_as_wav


def test_save_and_read(tmp_path):
    sr = 44100
    duration = 0.01  # 10 ms tone
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t, dtype=np.float32)

    out_file = tmp_path / "test_tone.wav"

    # Save without normalization so we can compare values closely
    save_sequence_as_wav(tone, sr, str(out_file), normalize=False, subtype="FLOAT")

    data, sr_read = sf.read(str(out_file), dtype="float32")

    assert sr_read == sr
    assert data.shape[0] == tone.shape[0]
    # The data should be nearly identical to the original when no normalization
    assert np.allclose(data, tone, atol=1e-6)


def test_soundsequence_save(tmp_path):
    """Ensure SoundSequence.save_to_wav writes the correct data."""
    from sound_generator import CreateSound, SoundSequence

    sr = 44100
    sound = CreateSound(freq=440, num_harmonics=1, tone_duration=0.01,
                        sample_rate=sr, harmonic_factor=1)
    seq_obj = SoundSequence(sound, total_duration=0.01, isi=0)

    out_file = "output/cond1.wav"

    # Save without normalization so read-back matches closely
    seq_obj.save_to_wav(str(out_file), normalize=False, subtype="FLOAT")

    data, sr_read = sf.read(str(out_file), dtype="float32")
    seq = seq_obj.generate_sequence()

    assert sr_read == sr
    assert data.shape[0] == seq.shape[0]
    assert np.allclose(data, seq, atol=1e-6)


seq_obj.save_to_wav("output/cond1.wav")