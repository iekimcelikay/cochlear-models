import numpy as np
import matplotlib.pyplot as plt
import math

# Load saved data
pure_tone = 1

if pure_tone:
    data = np.load('puretone_rf_data.npz')
else:
    data = np.load('hcomplextone_rf_data.npz')

rf = data['rf']  # shape (num_cf, num_dbs, num_freqs)
desired_frequencies = data['desired_frequencies']
desired_dbs = data['desired_dbs']
cf_list = data['cf_list']
num_cf = rf.shape[0]

#plot = 'single'
plot = 'multi'

if plot == 'single':

    # Plot heatmaps for channels, one by one
    for k in range(num_cf):
        rf_k = rf[k]  # shape: (num_dbs, num_freqs)

        fig, ax = plt.subplots(figsize=(6, 5))
        img = ax.imshow(rf_k, aspect='auto', origin='lower', cmap='viridis')

        ax.set_xticks(np.arange(len(desired_frequencies)))
        ax.set_xticklabels(["%.0f" % f for f in desired_frequencies])
        ax.set_yticks(np.arange(len(desired_dbs)))
        ax.set_yticklabels(["%d" % db for db in desired_dbs])

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Loudness (dB SPL)")
        title = "Receptive Field, Channel %d, freq= (CF = %.1f Hz)" % (k+1, cf_list[k])
        ax.set_title(title)

        cbar = fig.colorbar(img)
        cbar.set_label("Mean Firing Rate")

        plt.tight_layout()
        # Save the figure before showing
        rf_fname = "puretone_receptive_field_channel_%02d.png" % (k+1)
        plt.savefig(rf_fname)
        plt.show()

elif plot == 'multi':
    # Plot all channels in one figure (reuse the previous plotting code)
    num_rows = int(math.ceil(num_cf / 5))
    num_cols = min(num_cf, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3*num_rows), squeeze=False)

    for k in range(num_cf):
        row = k // num_cols
        col = k % num_cols

        ax = axes[row, col]
        rf_k = rf[k]

        img = ax.imshow(rf_k, aspect='auto', origin='lower', cmap='viridis')

        ax.set_xticks(np.arange(len(desired_frequencies)))
        ax.set_xticklabels(["%.0f" % f for f in desired_frequencies], rotation=70, fontsize=6)
        ax.set_yticks(np.arange(len(desired_dbs)))
        ax.set_yticklabels(["%d" % db for db in desired_dbs], fontsize=8)

        ax.set_xlabel("Freq (Hz)", fontsize=8, labelpad=0.8)
        ax.set_ylabel("Loudness (dB SPL)", fontsize=8, labelpad=0.5)

        ax.set_title("Ch %d (CF=%.1f Hz)" % (k+1, cf_list[k]), fontsize=11, pad=10, y=0.98)

    # Remove unused axes
    for idx in range(num_cf, num_rows*num_cols):
        fig.delaxes(axes.flatten()[idx])

    plt.subplots_adjust(
        left=0.06,
        right=0.87,
        top=0.93,
        bottom=0.07,
        wspace=0.3,
        hspace=0.4
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    fig.colorbar(img, cax=cbar_ax, label='Mean Firing Rate')

    plt.suptitle("Receptive Fields for All Channels - Pure Tone", fontsize=16, y=0.98)
    plt.show()

    # Save the figure
    fig.savefig("puretone_receptive_fields_all_channels.png", dpi=300, bbox_inches='tight')