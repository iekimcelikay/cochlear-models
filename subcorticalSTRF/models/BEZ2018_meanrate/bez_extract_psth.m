%% Extract PSTHs from BEZ2018 model
% This script extracts and saves PSTHs from BEZ2018 model files
% Run this script first to extract the data

clear; clc;

%% SETTINGS
folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/results";
n_runs = 10;
num_cf = 20;
min_cf = 125;
max_cf = 2500;
% psth_batch_10runs_20cfs_min125hz_max2500hz_128-128-128fibers

cfs = {'125.0', '159.49726883', '198.39318028', '242.24859198', '291.69587485', '347.44803149', ...
    '410.30897734', '481.18513264', '561.09849255', '651.20136376', '752.79298009', '867.33823676', ...
    '996.48881333', '1142.10699008', '1306.29250097', '1491.41281075', ...
    '1700.13725242', '1935.4755176', '2200.82105458', '2500.0'};
frequencies = cfs;
dbs = [50, 60, 70, 80];
fs = 100000;
target_fs = 1000;

fiber_group = '128-128-128'; % only one
folder_name = sprintf('psth_batch_%druns_%dcfs_min%dhz_max%dhz_%sfibers', n_runs, num_cf, min_cf, max_cf, fiber_group);
folder_path = fullfile(folder, folder_name);

% Create output directory if it doesn't exist
output_dir = fullfile(folder, 'processed_data');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Preallocate cell arrays for LSR, MSR, HSR
lsr_all = cell(num_cf, length(frequencies), length(dbs), n_runs);
msr_all = lsr_all;
hsr_all = lsr_all;
empty_cells = [];  % each row: [CF_idx, Tone_idx, dB_idx, Run_idx]

fprintf('Processing fiber group: %s\n', fiber_group);

%% LOAD DATA
for run_i = 0:n_runs-1
    for i_cf = 1:num_cf
        cf_val_num = str2double(cfs{i_cf});
        for i_tone = 1:length(frequencies)
            tone_val_num = str2double(frequencies{i_tone});
            for i_db = 1:length(dbs)
                db_val = dbs(i_db);
                filename = sprintf('synapse_result_cf%.1fhz_tone%.1fhz_%ddb_%sfibers_run%d.mat', ...
                                   cf_val_num, tone_val_num, db_val, fiber_group, run_i);
                filepath = fullfile(folder_path, filename);
                fprintf(' Loading %s\n', filename);
                
                try
                    data = load(filepath);
                    results = data.synapse_result_struct;

                    % --- DEBUG: check if PSTH is empty and fiber types present ---
                    if isempty(results.psth)
                        fprintf('WARNING: Empty PSTH at CF %g, Tone %g, dB %d, Run %d\n', ...
                                cf_val_num, tone_val_num, db_val, run_i+1);
                        % store indices instead of printing
                        empty_cells(end+1,:) = [i_cf, i_tone, i_db, run_i+1];
                        continue;
                    end
                    
                    unique_types = unique(results.anf_type);
                    fprintf('Fiber types in file: %s\n', strjoin(unique_types(:)', ', '));
                    
                    % Also check if any fiber types are missing
                    fiber_types = {'lsr','msr','hsr'};
                    missing_types = false;
                    for ft = fiber_types
                        if ~any(strcmp(results.anf_type, ft{1}))
                            % store as negative run index to indicate missing fiber type
                            fprintf('WARNING: Missing fiber type %s at CF %g, Tone %g, dB %d, Run %d\n', ...
                                    ft{1}, cf_val_num, tone_val_num, db_val, run_i+1);
                            empty_cells(end+1,:) = [i_cf, i_tone, i_db, run_i+1];
                            missing_types = true;
                        end
                    end
                    
                    if missing_types
                        continue;
                    end
                    
                    % --- LSR
                    lsr_idx = strcmp(results.anf_type, 'lsr');
                    lsr_spikes = results.psth(lsr_idx);
                    lsr_psths = cellfun(@(spk) compute_psth_resampled(double(spk), fs, target_fs), ...
                                        lsr_spikes, 'UniformOutput', false);
                    lsr_mat = cell2mat(cellfun(@(x) x(:), lsr_psths, 'UniformOutput', false));
                    lsr_all{i_cf, i_tone, i_db, run_i+1} = mean(lsr_mat, 2);

                    % --- MSR
                    msr_idx = strcmp(results.anf_type, 'msr');
                    msr_spikes = results.psth(msr_idx);
                    msr_psths = cellfun(@(spk) compute_psth_resampled(double(spk), fs, target_fs), ...
                                        msr_spikes, 'UniformOutput', false);
                    msr_mat = cell2mat(cellfun(@(x) x(:), msr_psths, 'UniformOutput', false));
                    msr_all{i_cf, i_tone, i_db, run_i+1} = mean(msr_mat, 2);

                    % --- HSR
                    hsr_idx = strcmp(results.anf_type, 'hsr');
                    hsr_spikes = results.psth(hsr_idx);
                    hsr_psths = cellfun(@(spk) compute_psth_resampled(double(spk), fs, target_fs), ...
                                        hsr_spikes, 'UniformOutput', false);
                    hsr_mat = cell2mat(cellfun(@(x) x(:), hsr_psths, 'UniformOutput', false));
                    hsr_all{i_cf, i_tone, i_db, run_i+1} = mean(hsr_mat, 2);
                    
                catch e
                    fprintf('ERROR loading file %s: %s\n', filename, e.message);
                    empty_cells(end+1,:) = [i_cf, i_tone, i_db, run_i+1];
                end
            end
        end
    end
end

% --- DEBUG: check which CFs actually have data ---
for i_cf = 1:num_cf
    filled_lsr = sum(~cellfun(@isempty, squeeze(lsr_all(i_cf,:,:,:))), 'all');
    filled_msr = sum(~cellfun(@isempty, squeeze(msr_all(i_cf,:,:,:))), 'all');
    filled_hsr = sum(~cellfun(@isempty, squeeze(hsr_all(i_cf,:,:,:))), 'all');
    fprintf('CF %d: LSR %d entries, MSR %d, HSR %d\n', i_cf, filled_lsr, filled_msr, filled_hsr);
end

if isempty(empty_cells)
    fprintf('All PSTHs loaded successfully. No empty cells.\n');
else
    fprintf('WARNING: Empty or missing PSTHs detected in %d entries:\n', size(empty_cells,1));
    fprintf('CF_idx\tTone_idx\tdB_idx\tRun_idx\n');
    disp(empty_cells);
end

% Save extracted PSTHs to file
save(fullfile(output_dir, 'psth_data_128fibers.mat'), 'lsr_all', 'msr_all', 'hsr_all', 'cfs', 'frequencies', 'dbs', 'target_fs', 'n_runs', 'empty_cells');
fprintf('Saved PSTH data to %s\n', fullfile(output_dir, 'psth_data_128fibers.mat'));