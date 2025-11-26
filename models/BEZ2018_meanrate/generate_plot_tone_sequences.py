#!/usr/bin/env python3
"""
Pure Tone Sequence Generator and Visualization Script

This script generates 9 different pure tone sequences with varying tone durations and 
inter-stimulus intervals (ISI), then creates visualizations for each condition.

Conditions:
1. 133 ms tone, 33 ms ISI
2. 133 ms tone, 200 ms ISI
3. 133 ms tone, 867 ms ISI
4. 33 ms tone, 133 ms ISI
5. 200 ms tone, 133 ms ISI
6. 867 ms tone, 133 ms ISI
7. 267 ms tone, 67 ms ISI
8. 800 ms tone, 200 ms ISI
9. 5000 ms tone, 0 ms ISI

Author: [Generated for subcorticalSTRF project]
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os
import signal
import sys
from soundgen import SoundGen

# Configure matplotlib to use non-interactive backend if needed
plt.ion()  # Turn on interactive mode

# sounddevice is optional for audio playback
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    print("Note: sounddevice not available. Audio playback disabled, but visualization will work.")

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print('\nInterrupted by user. Exiting gracefully...')
    plt.close('all')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def create_output_directory():
    """Create output directory for saving plots and audio files."""
    output_dir = "tone_sequence_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_sequence(sequence, sample_rate, condition_num, tone_duration, isi, output_dir, 
                 freq=100, num_harmonics=4, total_duration=5.0):
    """
    Plot and save a tone sequence visualization.
    
    Parameters:
    -----------
    sequence : numpy.ndarray
        The audio sequence to plot
    sample_rate : float
        Sample rate in Hz
    condition_num : int
        Condition number (1-9)
    tone_duration : float
        Duration of each tone in seconds
    isi : float
        Inter-stimulus interval in seconds
    output_dir : str
        Directory to save plots
    freq : float
        Base frequency in Hz
    num_harmonics : int
        Number of harmonics
    total_duration : float
        Total sequence duration in seconds
    """
    
    # Create time axis in milliseconds
    t_ms = np.linspace(0, len(sequence) / sample_rate * 1000, len(sequence))
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Main waveform plot
    plt.subplot(2, 1, 1)
    plt.plot(t_ms, sequence, color='darkblue', linewidth=0.8)
    plt.title(f'Condition {condition_num}: Tone Duration = {tone_duration*1000:.0f} ms, ISI = {isi*1000:.0f} ms\n'
              f'Pure Tone: Frequency = {freq} Hz, Total Duration = {total_duration} s', 
              fontsize=12, fontweight='bold')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Zoomed view of first few seconds
    plt.subplot(2, 1, 2)
    zoom_duration_ms = min(2000, len(sequence) / sample_rate * 1000)  # Show first 2 seconds or less
    zoom_samples = int(zoom_duration_ms * sample_rate / 1000)
    plt.plot(t_ms[:zoom_samples], sequence[:zoom_samples], color='red', linewidth=1.2)
    plt.title(f'Zoomed View - First {zoom_duration_ms/1000:.1f} seconds', fontsize=10)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'condition_{condition_num:02d}_tone_{tone_duration*1000:.0f}ms_isi_{isi*1000:.0f}ms.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show(block=False)  # Non-blocking show
    plt.pause(0.1)  # Brief pause to render
    
    print(f"Saved plot: {plot_path}")

def save_audio(sequence, sample_rate, condition_num, tone_duration, isi, output_dir):
    """Save audio sequence as WAV file."""
    audio_filename = f'condition_{condition_num:02d}_tone_{tone_duration*1000:.0f}ms_isi_{isi*1000:.0f}ms.wav'
    audio_path = os.path.join(output_dir, audio_filename)
    
    # Normalize to prevent clipping and convert to 16-bit
    normalized_sequence = sequence / np.max(np.abs(sequence)) * 0.8
    audio_int16 = (normalized_sequence * 32767).astype(np.int16)
    
    write(audio_path, int(sample_rate), audio_int16)
    print(f"Saved audio: {audio_path}")

def generate_all_tone_sequences():
    """Generate and visualize all 9 tone sequence conditions."""
    
    # Initialize the sound generator
    sample_rate = 100e3  # 100 kHz sample rate
    tau = 0.01  # 10 ms ramping window
    sound_gen = SoundGen(sample_rate, tau)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Define sequence parameters for all 9 conditions
    # Format: (freq, num_harmonics, tone_duration, harmonic_factor, dbspl, total_duration, isi)
    sequence_params = [
        (100, 1, 0.133, 0.5, 60, 5.0, 0.033),  # Condition 1: 133 ms tone, 33 ms ISI
        (100, 1, 0.133, 0.5, 60, 5.0, 0.200),  # Condition 2: 133 ms tone, 200 ms ISI
        (100, 1, 0.133, 0.5, 60, 5.0, 0.867),  # Condition 3: 133 ms tone, 867 ms ISI
        (100, 1, 0.033, 0.5, 60, 5.0, 0.133),  # Condition 4: 33 ms tone, 133 ms ISI
        (100, 1, 0.200, 0.5, 60, 5.0, 0.133),  # Condition 5: 200 ms tone, 133 ms ISI
        (100, 1, 0.867, 0.5, 60, 5.0, 0.133),  # Condition 6: 867 ms tone, 133 ms ISI
        (100, 1, 0.267, 0.5, 60, 5.0, 0.067),  # Condition 7: 267 ms tone, 67 ms ISI
        (100, 1, 0.800, 0.5, 60, 5.0, 0.200),  # Condition 8: 800 ms tone, 200 ms ISI
        (100, 1, 5.000, 0.1, 60, 5.0, 0.000),  # Condition 9: 5000 ms tone, 0 ms ISI
    ]
    
    # Generate all sequences
    sequences = []
    condition_info = []
    
    print("Generating tone sequences...")
    print("=" * 60)
    
    for i, params in enumerate(sequence_params, 1):
        freq, num_harmonics, tone_duration, harmonic_factor, dbspl, total_duration, isi = params
        
        print(f"Condition {i}: Generating tone (duration: {tone_duration*1000:.0f} ms, ISI: {isi*1000:.0f} ms)")
        
        # Generate the sequence
        sequence = sound_gen.generate_sequence(freq, num_harmonics, tone_duration, 
                                             harmonic_factor, dbspl, total_duration, isi)
        sequences.append(sequence)
        condition_info.append({
            'condition': i,
            'tone_duration': tone_duration,
            'isi': isi,
            'freq': freq,
            'num_harmonics': num_harmonics,
            'total_duration': total_duration
        })
        
        # Create visualization
        plot_sequence(sequence, sample_rate, i, tone_duration, isi, output_dir, 
                     freq, num_harmonics, total_duration)
        
        # Save audio file
        save_audio(sequence, sample_rate, i, tone_duration, isi, output_dir)
        
        print(f"  - Sequence length: {len(sequence)} samples ({len(sequence)/sample_rate:.2f} seconds)")
        print()
    
    # Create summary plot showing all conditions
    create_summary_plot(sequences, sample_rate, condition_info, output_dir)
    
    # Create timing analysis plot
    create_timing_analysis_plot(condition_info, output_dir)
    
    print("=" * 60)
    print(f"All sequences generated and saved to: {output_dir}")
    print(f"Total conditions: {len(sequences)}")
    
    return sequences, condition_info, output_dir

def create_summary_plot(sequences, sample_rate, condition_info, output_dir):
    """Create a summary plot showing all conditions."""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, (sequence, info) in enumerate(zip(sequences, condition_info)):
        ax = axes[i]
        
        # Show first 2 seconds or full sequence if shorter
        max_time_ms = min(2000, len(sequence) / sample_rate * 1000)
        max_samples = int(max_time_ms * sample_rate / 1000)
        
        t_ms = np.linspace(0, max_time_ms, max_samples)
        ax.plot(t_ms, sequence[:max_samples], color=f'C{i}', linewidth=0.8)
        
        ax.set_title(f'Condition {info["condition"]}\n'
                    f'Tone: {info["tone_duration"]*1000:.0f} ms, '
                    f'ISI: {info["isi"]*1000:.0f} ms', fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits
        ax.set_ylim(-1, 1)
    
    plt.suptitle('Pure Tone Sequences - All Conditions Overview\n(First 2 seconds shown)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'all_conditions_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.show(block=False)  # Non-blocking show
    plt.pause(0.1)  # Brief pause to render
    
    print(f"Saved summary plot: {summary_path}")

def create_timing_analysis_plot(condition_info, output_dir):
    """Create a timing analysis visualization."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    conditions = [info['condition'] for info in condition_info]
    tone_durations = [info['tone_duration'] * 1000 for info in condition_info]  # Convert to ms
    isis = [info['isi'] * 1000 for info in condition_info]  # Convert to ms
    
    # Bar plot of tone durations
    bars1 = ax1.bar(conditions, tone_durations, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Condition Number')
    ax1.set_ylabel('Tone Duration (ms)')
    ax1.set_title('Tone Duration by Condition')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization of large range
    
    # Add value labels on bars
    for bar, duration in zip(bars1, tone_durations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{duration:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Bar plot of ISI durations
    bars2 = ax2.bar(conditions, isis, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('Condition Number')
    ax2.set_ylabel('Inter-Stimulus Interval (ms)')
    ax2.set_title('ISI Duration by Condition')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    # Add value labels on bars
    for bar, isi in zip(bars2, isis):
        height = bar.get_height()
        if height > 0:  # Only add label if ISI > 0
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{isi:.0f}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., 1,
                    '0', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Timing Parameters Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timing_path = os.path.join(output_dir, 'timing_analysis.png')
    plt.savefig(timing_path, dpi=300, bbox_inches='tight')
    plt.show(block=False)  # Non-blocking show
    plt.pause(0.1)  # Brief pause to render
    
    print(f"Saved timing analysis: {timing_path}")

def play_sequence_demo(sequences, sample_rate, condition_num=1):
    """
    Play a specific sequence for demonstration.
    
    Parameters:
    -----------
    sequences : list
        List of generated sequences
    sample_rate : float
        Sample rate in Hz
    condition_num : int
        Which condition to play (1-9)
    """
    if not AUDIO_PLAYBACK_AVAILABLE:
        print("Audio playback not available (sounddevice not installed)")
        return
        
    if 1 <= condition_num <= len(sequences):
        print(f"Playing Condition {condition_num}...")
        sd.play(sequences[condition_num-1], samplerate=sample_rate)
        sd.wait()
        print("Playback complete.")
    else:
        print(f"Invalid condition number. Choose between 1 and {len(sequences)}.")

if __name__ == "__main__":
    print("Pure Tone Sequence Generator")
    print("=" * 60)
    print("This script generates 9 different pure tone sequences with")
    print("varying tone durations and inter-stimulus intervals (ISI).")
    print("Press Ctrl+C to interrupt and exit at any time.")
    print()
    
    try:
        # Generate all sequences
        sequences, condition_info, output_dir = generate_all_tone_sequences()
    
    # Optional: Play a demonstration (uncomment to enable)
    # print("\nWould you like to hear a demonstration of Condition 1? (y/n)")
    # response = input().strip().lower()
    # if response == 'y' or response == 'yes':
    #     play_sequence_demo(sequences, 100e3, 1)
    
        print("\nScript completed successfully!")
        print(f"Check the '{output_dir}' directory for all generated files.")
        
        # Keep plots open for viewing
        print("\nPlots are displayed. Close plot windows or press Ctrl+C to exit.")
        plt.show()  # Keep plots open
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        plt.close('all')
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        plt.close('all')
        sys.exit(1)