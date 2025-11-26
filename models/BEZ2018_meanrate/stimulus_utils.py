# stimulus_utils.py
import numpy as np

from BEZ2018_meanrate.simulator_utils import calc_cfs

def generate_stimuli_params(freq_range, db_range):
    """
    Generate stimulus parameters for frequencies and dB levels.
    
    Args:
        freq_range: Tuple of (min_freq, max_freq, num_freqs) for frequency range
        db_range: Either a tuple (min_db, max_db, step) for range, or a single number for one dB level
        
    Returns:
        desired_dbs: Array of dB levels
        desired_freqs: Array of frequencies
    """
    if np.isscalar(db_range):
        desired_dbs = np.array([db_range])
    else:
        desired_dbs = np.arange(*db_range)
    
    desired_freqs = calc_cfs(freq_range, species='human')
    return desired_dbs, desired_freqs

def generate_ramped_tone(sound_gen, freq, num_harmonics, duration, harmonic_factor, db):
    tone = sound_gen.sound_maker(freq, num_harmonics, duration, harmonic_factor, db)
    ramped_tone = sound_gen.sine_ramp(tone)
    return ramped_tone


def generate_tone_dictionary(
    sound_gen, db_range, freq_range,
    num_tones, num_harmonics, duration, harmonic_factor
):
    desired_dbs, desired_freqs = generate_stimuli_params(freq_range, num_tones, db_range)
    return {
        (db, freq): generate_ramped_tone(
            sound_gen, freq, num_harmonics, duration, harmonic_factor, db
        )
        for db in desired_dbs for freq in desired_freqs
    }

def generate_tone_generator(
        sound_gen, db_range, freq_range, num_harmonics, duration, harmonic_factor):
    """
    Generate tones on-the-fly using a generator.
    
    Args:
        sound_gen: SoundGen instance
        db_range: Either a tuple (min_db, max_db, step) or a single dB value
        freq_range: Tuple of (min_freq, max_freq, num_freqs)
        num_harmonics: Number of harmonics in the tone
        duration: Duration of tone in seconds
        harmonic_factor: Harmonic amplitude decay factor
        
    Yields:
        Tuple of (db, freq, tone)
    """
    desired_dbs, desired_freqs = generate_stimuli_params(freq_range, db_range)
    for db in desired_dbs:
        for freq in desired_freqs:
            # Generate tone only when this iteration happens 
            tone = generate_ramped_tone(sound_gen, freq, num_harmonics, duration, harmonic_factor, db)
            # Yield tuple of info and tone, so the caller receives it 
            yield (db, freq, tone)


#----------------------------------------

