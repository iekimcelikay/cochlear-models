function [spike_rates, time_axis] = compute_psth_resampled(spikes, fs, target_fs)
    % Bin width in seconds
    bin_width_s = 1 / target_fs;

    % No of samples per bin in the original signal
    bin_samples = round(fs * bin_width_s);

    % No of full bins that fit into the spike train
    n_bins = floor(length(spikes) / bin_samples);

    spike_rates = zeros(1, n_bins);

    for i = 1:n_bins
        start_idx = (i-1)*bin_samples + 1;
        end_idx = i * bin_samples;

        bin_spikes = spikes(start_idx:end_idx);
        spike_count = sum(bin_spikes);

        spike_rates(i) = spike_count / bin_width_s; % Spike rate in Hz
    end

    time_axis = (0:n_bins-1) * bin_width_s;

end
