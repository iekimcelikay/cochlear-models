# This will be the functional script adapted from sound_generator.py
#import cochlea
import numpy as np
import math
import thorns
import time
from thorns import waves as wv
import scipy.signal as signal
from scipy.signal import firwin, filtfilt

# This script will define the functions, and then another module script will call these functions.

class SoundGen:

    def __init__(self, sample_rate, tau):
        """
        Initialize the CreateSound instance.
        :param sample_rate: Sample rate of sounds ( per second).
        :param tau: The ramping window in seconds.
        """
        self.sample_rate = sample_rate
        self.tau = tau


    def sound_maker(self, freq, num_harmonics, tone_duration, harmonic_factor):
        """
        :param freq: Base frequency in Hz.
        :param num_harmonics: Number of harmonic tones.
        :param tone_duration: Duration of each tone in seconds.
        :param harmonic_factor: Harmonic amplitude decay factor for each tone.
        :param dbspl: Desired dbspl (loudness) level.
        :return: sound: np.ndarray: array of audio samples representing the harmonic complex tone.
        """
        # Create the time array
        t = np.linspace(0, tone_duration, int(self.sample_rate * tone_duration), endpoint=False)
        # Initialize the sound array
        sound = np.zeros_like(t)
        # Generate the harmonics
        for k in range(1, num_harmonics + 1):
            omega = 2 * np.pi * freq * k
            harmonic = np.sin(omega * t)
            amplitude = harmonic_factor ** (k - 1)
            sound = sound + amplitude * harmonic
        
        # TODO: call thorns to normalize to a certain dbspl level 
        normalized_sound = thorns.waves.set_dbspl(sound, 60)  # Example: set to 70 dB SPL
        max_amplitude = 0.2753 # found this value in simulation, `maxamp_simulation.py`
        normalized_sound = normalized_sound / (max_amplitude + 0.01)  
        
        # max_amplitude = 1.27 # TODO: calculate this number through the for loop simulation and it should be the same for every sound.
        # sound = sound / (max_amplitude + 0.01)  # Scale down only if needed

        return normalized_sound

    def noise_maker(self, tone_duration, seed=None):
        if seed is not None:
            np.random.seed(seed)
        sound = np.random.normal(0, 1, tone_duration * self.sample_rate)
        return sound

    def bandpass_filter_FIR(self, tone_duration, dbspl, lowcut, highcut, numtaps, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Generate the white noise using noise_maker method
        noise = self.noise_maker(tone_duration)
        # design FIR bandpass filter
        fir_coeffs = firwin(numtaps, [lowcut, highcut], pass_zero=True, fs = self.sample_rate)

        # apply zero-phase FIR filter (no delay)
        filtered_noise = filtfilt(fir_coeffs, [1.0], noise)

        # set to desired dbspl using thorns
        normalized_noise = thorns.waves.set_dbspl(filtered_noise, dbspl)
        return normalized_noise

    def generate_multiple_band_limited_noises(self, n_trials, tone_duration, lowcut, highcut, numtaps,dbspl,base_seed=0):
        """
        Generate multiple reproducible band-limited white noise stimuli for the Zilany model.

        Parameters:
            n_trials (int): Number of stimuli to generate
            tone_duration (float): Duration in seconds
            lowcut (float): Lower cutoff frequency of bandpass filter (Hz)
            highcut (float): Upper cutoff frequency of bandpass filter (Hz)
            numtaps (int): FIR filter length
            dbspl (float): Desired dB SPL
            base_seed (int): Seed offset to allow reproducible variation

        Returns:
            list of numpy.ndarray: List of filtered, level-adjusted noise signals
        """
        fir_coeffs = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=self.sample_rate)
        n_samples = int(tone_duration * self.sample_rate)
        stimuli = []

        for i in range(n_trials):
            seed = base_seed + i
            np.random.seed(seed)
            white_noise = np.random.normal(0, 1, n_samples)
            filtered = filtfilt(fir_coeffs, [1.0], white_noise)
            scaled = thorns.waves.set_dbspl(filtered, dbspl)
            stimuli.append(scaled)

        return stimuli

    def ramp_in_out(self, sound):
        sound = sound.copy()
        L = int (self.tau * self.sample_rate)
        hw = np.hamming(2*L)

        sound = sound.copy()
        sound[:L] *= hw[:L]
        sound[-L:] *= hw[L:]
        return sound

    def sine_ramp(self, sound):
        L = int(self.tau * self.sample_rate)
        t = np.linspace(0, L / self.sample_rate, L)
        sine_window = np.sin(np.pi * t / (2 * self.tau)) ** 2  # Sine fade-in

        sound = sound.copy()
        sound[:L] *= sine_window  # Apply fade-in
        sound[-L:] *= sine_window[::-1]  # Apply fade-out

        return sound

    def generate_sequence(self, freq, num_harmonics, tone_duration, harmonic_factor,  total_duration, isi, stereo=True):

        # Generate the tone using the sound_maker method
        sound = self.sound_maker(freq,num_harmonics, tone_duration, harmonic_factor)
        # Sine ramp
        ramped_sound = self.sine_ramp(sound)
        # Calculate the number of tones that can fit into the total duration
        total_samples = int(total_duration * self.sample_rate)
        isi_samples = int(isi * self.sample_rate)
        num_tones = int((total_samples + isi_samples) // (len(ramped_sound) + isi_samples))

        # Generate the sequence with ISI gaps between each tone
        sequence = np.array([])

        for _ in range(num_tones):
            sequence = np.concatenate((sequence, ramped_sound))

            # Add ISI (silent gap) between tones
            sequence = np.concatenate((sequence, np.zeros(isi_samples)))

        # Ensure the sequence doesn't exceed the total duration
        if len(sequence) > total_samples:
            sequence = sequence[:total_samples]

        # If stereo is desired, duplicate the mono sequence into two channels
        if stereo:
            sequence = np.column_stack((sequence, sequence))

        return sequence
    
    def sample_frequencies_gaussian(self, freq_mean, freq_std, num_samples, freq_min=None, freq_max=None, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Sample from Gaussian distribution
        frequencies = np.random.normal(freq_mean, freq_std, num_samples)

        # Apply min/max constraints if provided
        if freq_min is not None and freq_max is not None:
            frequencies = np.clip(frequencies, freq_min, freq_max)
        elif freq_min is not None:
            frequencies = np.maximum(frequencies, freq_min)
        elif freq_max is not None:
            frequencies = np.minimum(frequencies, freq_max)
        
        # Ensure no negative frequencies (physically meaningless)
        frequencies = np.abs(frequencies)

        return frequencies
    
    def calculate_num_tones(self, tone_duration, isi, total_duration):
        total_samples = int(total_duration * self.sample_rate)
        isi_samples = int(isi * self.sample_rate)
        tone_samples = int(tone_duration * self.sample_rate)
        num_tones = int((total_samples + isi_samples) // (tone_samples + isi_samples))
        return num_tones
    
    def generate_sequence_from_freq_array(self, frequencies, num_harmonics, 
                                          tone_duration, harmonic_factor,
                                          isi, total_duration=None, 
                                          stereo=True):
        isi_samples = int(isi * self.sample_rate)
        sequence = np.array([])

         # Generate each tone at its specified frequency 
        for freq in frequencies:
            # Generate tone at this frequency
            sound = self.sound_maker(freq, num_harmonics, tone_duration,
                                        harmonic_factor)
            
             # Apply ramping (I will try hamming this time too) 
            ramped_sound = self.sine_ramp(sound)

             # Add tone to sequence
            sequence = np.concatenate((sequence, ramped_sound))

             # Add ISI (silence) after tone 
            sequence = np.concatenate((sequence, np.zeros(isi_samples)))

        # Trim to exact total duration if specified
        if total_duration is not None:
                total_samples = int(total_duration * self.sample_rate)
                if len(sequence) > total_samples:
                    sequence = sequence[:total_samples] 

        # If stereo is desired, duplicate the mono sequence into two channels
        if stereo:
            sequence = np.column_stack((sequence, sequence))
        return sequence

    def generate_sequence_gaussian_freq(self, freq_mean, freq_std, num_harmonics, 
                                       tone_duration, harmonic_factor, 
                                       total_duration, isi, 
                                       freq_min=None, freq_max=None, seed=None,
                                       stereo=True):
        
        """
        Convenience method: Generate sequence with Gaussian-distributed frequencies.
        
        This combines the modular methods above into one call.
        :param freq_mean: Mean frequency for Gaussian sampling.
        :param freq_std: Standard deviation for Gaussian sampling.
        :param num_harmonics: Number of harmonics for each tone.
        :param tone_duration: Duration of each tone in seconds.
        :param harmonic_factor: Amplitude decay factor for harmonics.
        :param total_duration: Total duration of the sequence in seconds.
        :param isi: Inter-stimulus interval in seconds.
        :param freq_min: Minimum frequency constraint (optional).
        :param freq_max: Maximum frequency constraint (optional).
        :param seed: Random seed for reproducibility (optional).
        :param stereo: Whether to generate stereo output (default True).

        :return: tuple: (sequence, frequencies)
            - sequence: np.ndarray: Generated sound sequence.
            - frequencies: np.ndarray: Array of sampled frequencies used in the sequence.
            
        """
        # Step 1: Calculate how many tones we need
        num_tones = self.calculate_num_tones(tone_duration, isi, total_duration)
        
        # Step 2: Sample frequencies
        frequencies = self.sample_frequencies_gaussian(
            freq_mean, freq_std, num_tones, freq_min, freq_max, seed
        )
        
        # Step 3: Generate sequence from frequency array
        sequence = self.generate_sequence_from_freq_array(
            frequencies, num_harmonics, tone_duration, harmonic_factor, 
             isi, total_duration, stereo=stereo
        )
        
        return sequence, frequencies
    
    
    
    ## TODO: Move this simulators to a different script file later.
###
    def peripheral_simulator(self, sequence, peripheral_fs, num_cf):
        """
        :param sequence: array_like: Sound sequence as np array
        :param peripheral_fs: floatL Sampling frequency of the peripheral signal (diff. from original fs).
        anf_num: tuple: The desired number of auditory nerve fibers per frequency channel (CF), (HSR#, MSR#, LSR#).
        cf: tuple: (min_cf, max_cf, num_cf).
        :param num_cf: number of frequency channels.
        species: {'cat', 'human', 'human_glasberg1990'}
        seed: int: Random seed for the spike generator.
        :return: anf_trains: Auditory nerve spike trains.
        """
        if self.sample_rate != peripheral_fs:
            sequence = wv.resample(sequence, self.sample_rate, peripheral_fs)
        anf_trains = cochlea.run_zilany2014(sequence, peripheral_fs,
                                            anf_num = (60, 25, 15),
                                            cf = (125, 2500, num_cf),
                                            species= 'human', seed = 557)

        return anf_trains
    def peripheral_sim_rate(self, sequence, peripheral_fs, num_cf, subcortical_fs):
        if self.sample_rate!= peripheral_fs:
            sequence = wv.resample(sequence, self.sample_rate, peripheral_fs)

        np.random.seed()
        auditory_nerve_rate_ts = cochlea.run_zilany2014_rate(sequence, peripheral_fs,
                                                             anf_types= ('lsr', 'msr', 'hsr'),
                                                             cf = (125, 2500, num_cf),
                                                             species= 'human',
                                                             cohc = 1,
                                                             cihc = 1,
                                                             powerlaw = 'actual',
                                                             ffGn = True)
        #auditory_nerve_rate_ts = .6 * auditory_nerve_rate_ts['hsr'] + .25 * auditory_nerve_rate_ts['msr'] + .15 * auditory_nerve_rate_ts['lsr']
        #resampleN = len(sequence) * subcortical_fs / peripheral_fs
        #p = wv.resample(auditory_nerve_rate_ts, peripheral_fs, subcortical_fs)  # signal = scipy.signal
        #p[p < 0] = 0
        #p[p > 1] = 1

        return auditory_nerve_rate_ts

    def cochlea_rate_manual(self, sequence, peripheral_fs, min_cf, max_cf,num_cf,  num_anf, seed= None):
        """
        num_ANF must be tuple.
        :param sequence:
        :param peripheral_fs:
        :param min_cf:
        :param max_cf:
        :param num_cf:
        :param num_anf:
        :return:
        """
        if seed is None:
            seed = int(time.time() * 1e6) % (2**32 - 1)

        print(seed)


        if self.sample_rate!= peripheral_fs:
            sequence = wv.resample(sequence, self.sample_rate, peripheral_fs)
        anf_trains = cochlea.run_zilany2014(sequence, peripheral_fs,
                                                anf_num=num_anf,
                                                cf=(min_cf, max_cf, num_cf),
                                                species='human', seed=seed, cohc=1, cihc=1, powerlaw='actual', ffGn=True)
        # Accumulate the spike trainsreturn anf_trains
        #anf_acc = thorns.accumulate(anf_trains, keep=['cf', 'type'])
        anf_acc = anf_trains

        return anf_acc

####
