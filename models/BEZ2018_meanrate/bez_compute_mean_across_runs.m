clear; clc;

% Load BEZ model data
bez_dir = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/results/processed_data";
bez_file = fullfile(bez_dir, 'psth_data_128fibers.mat');
assert(isfile(bez_file), 'BEZ model PSTH data not found. Run bez_extract_psth.m first.');

%% Compute the means of the runs of the BEZ model first
load(bez_file);

% Verify n_runs is available
if ~exist('n_runs', 'var')
    % If not available in the data file, determine it from the data dimensions
    [~, ~, ~, n_runs] = size(lsr_all);
    fprintf('n_runs not found in data file, determined from data dimensions: %d\n', n_runs);
end

% Get dimensions
fiber_types = {'lsr', 'msr', 'hsr'};
num_cf = length(cfs);
cf_vals = cellfun(@str2double, cfs);

fprintf('Loaded data for:\n');
fprintf('  %d CFs: %.1f to %.1f Hz\n', num_cf, min(cf_vals), max(cf_vals));
fprintf('  %d Tone freqs: %.1f to %.1f Hz\n', length(frequencies), min(str2double(frequencies{1})), max(str2double(frequencies{end})));
fprintf('  %d dB levels: %s\n', length(dbs), mat2str(dbs));
fprintf('  %d runs per condition\n', n_runs);

% Initialize output structures with same format as cochlea
% Each cell contains a 1x200 PSTH (average of 10 runs)
bez_rates = struct();
sem_rates = struct();

% Initialize cell arrays for each fiber type
bez_rates.lsr = cell(num_cf, length(frequencies), length(dbs));
bez_rates.msr = cell(num_cf, length(frequencies), length(dbs));
bez_rates.hsr = cell(num_cf, length(frequencies), length(dbs));
sem_rates.lsr = cell(num_cf, length(frequencies), length(dbs));
sem_rates.msr = cell(num_cf, length(frequencies), length(dbs));
sem_rates.hsr = cell(num_cf, length(frequencies), length(dbs));

% Process each condition
for i_cf = 1:num_cf
    for i_tone = 1:length(frequencies)
        for i_db = 1:length(dbs)
            % Process each fiber type
            for ft_idx = 1:length(fiber_types)
                ft = fiber_types{ft_idx};
                
                % Get all runs for this condition and fiber type
                if strcmp(ft, 'lsr')
                    all_runs = squeeze(lsr_all(i_cf, i_tone, i_db, :));
                elseif strcmp(ft, 'msr')
                    all_runs = squeeze(msr_all(i_cf, i_tone, i_db, :));
                elseif strcmp(ft, 'hsr')
                    all_runs = squeeze(hsr_all(i_cf, i_tone, i_db, :));
                end
                
                % Skip if empty
                if isempty(all_runs) || all(cellfun(@isempty, all_runs))
                    bez_rates.(ft){i_cf, i_tone, i_db} = [];
                    sem_rates.(ft){i_cf, i_tone, i_db} = [];
                    continue;
                end
                
                % Convert to matrix for averaging [timepoints × runs]
                valid_runs = find(~cellfun(@isempty, all_runs));
                if isempty(valid_runs)
                    bez_rates.(ft){i_cf, i_tone, i_db} = [];
                    sem_rates.(ft){i_cf, i_tone, i_db} = [];
                    continue;
                end
                
                % Stack valid PSTHs as columns in a matrix
                psth_matrix = [];
                for run_idx = 1:length(valid_runs)
                    run_data = all_runs{valid_runs(run_idx)};
                    
                    % Ensure consistent orientation (column vector)
                    if size(run_data, 1) < size(run_data, 2)
                        run_data = run_data';
                    end
                    
                    % Add to matrix
                    psth_matrix = [psth_matrix, run_data];
                end
                
                % Calculate mean and SEM across runs
                if ~isempty(psth_matrix)
                    mean_psth = mean(psth_matrix, 2, 'omitnan');
                    if size(valid_runs, 1) > 1
                        sem_psth = std(psth_matrix, 0, 2, 'omitnan') / sqrt(size(psth_matrix, 2));
                    else
                        sem_psth = zeros(size(mean_psth));
                    end
                    
                    % Store in output structure (as row vectors to match cochlea format)
                    bez_rates.(ft){i_cf, i_tone, i_db} = mean_psth';
                    sem_rates.(ft){i_cf, i_tone, i_db} = sem_psth';
                else
                    bez_rates.(ft){i_cf, i_tone, i_db} = [];
                    sem_rates.(ft){i_cf, i_tone, i_db} = [];
                end
            end
        end
    end
    fprintf('Processed CF %d/%d: %.1f Hz\n', i_cf, num_cf, cf_vals(i_cf));
end

% Count filled entries for each fiber type
lsr_filled = sum(~cellfun(@isempty, bez_rates.lsr(:)));
msr_filled = sum(~cellfun(@isempty, bez_rates.msr(:)));
hsr_filled = sum(~cellfun(@isempty, bez_rates.hsr(:)));
total_cells = numel(bez_rates.lsr);

fprintf('\nSummary:\n');
fprintf('LSR: %d/%d cells filled (%.1f%%)\n', lsr_filled, total_cells, 100*lsr_filled/total_cells);
fprintf('MSR: %d/%d cells filled (%.1f%%)\n', msr_filled, total_cells, 100*msr_filled/total_cells);
fprintf('HSR: %d/%d cells filled (%.1f%%)\n', hsr_filled, total_cells, 100*hsr_filled/total_cells);

% Check PSTH dimensions for consistency
fprintf('\nChecking PSTH dimensions:\n');
psth_sizes = cell(3,1);
if lsr_filled > 0
    non_empty = bez_rates.lsr(~cellfun(@isempty, bez_rates.lsr));
    size_strs = cellfun(@(x) sprintf('%dx%d', size(x,1), size(x,2)), non_empty, 'UniformOutput', false);
    psth_sizes{1} = unique(size_strs);
    fprintf('LSR PSTH dimensions: %s\n', strjoin(psth_sizes{1}, ', '));
end

if msr_filled > 0
    non_empty = bez_rates.msr(~cellfun(@isempty, bez_rates.msr));
    size_strs = cellfun(@(x) sprintf('%dx%d', size(x,1), size(x,2)), non_empty, 'UniformOutput', false);
    psth_sizes{2} = unique(size_strs);
    fprintf('MSR PSTH dimensions: %s\n', strjoin(psth_sizes{2}, ', '));
end

if hsr_filled > 0
    non_empty = bez_rates.hsr(~cellfun(@isempty, bez_rates.hsr));
    size_strs = cellfun(@(x) sprintf('%dx%d', size(x,1), size(x,2)), non_empty, 'UniformOutput', false);
    psth_sizes{3} = unique(size_strs);
    fprintf('HSR PSTH dimensions: %s\n', strjoin(psth_sizes{3}, ', '));
end

% Save the computed rates [1x200] 
save(fullfile(bez_dir, 'bez_acrossruns_psths.mat'), 'bez_rates', 'sem_rates', ...
     'cfs', 'frequencies', 'dbs', 'n_runs', '-v7');
fprintf('\nComputed mean PSTHs across %d runs for BEZ model and saved to bez_acrossruns_psths.mat\n', n_runs);
