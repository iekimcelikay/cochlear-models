% filepath: /home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate/aggregate_cochlea_psths.m
%% Aggregate existing PSTH files into a single file for comparison with BEZ model
% This script combines existing PSTHs and organizes them to match BEZ model structure
% No SEM calculation (single run)

% WORKS WELL. Run after generating cochlea model PSTHs with compute_cochlea_psths.m

clear; clc;

% Input and output directories
input_dir = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate/out/condition_psths";
output_file = fullfile(input_dir, 'cochlea_psths.mat');

% Define fiber types
fiber_types = {'lsr', 'msr', 'hsr'};

% List condition files with the correct pattern
file_pattern = fullfile(input_dir, 'psth_cf*_tone*_db*.mat');
files = dir(file_pattern);

if isempty(files)
    error('No condition files found matching the pattern %s', file_pattern);
end

fprintf('Found %d condition files to process\n', length(files));

% First pass: identify all unique parameters
expr = '^psth_cf(?<cf>[-+]?\d*\.?\d+([eE][-+]?\d+)?)_tone(?<tone>[-+]?\d*\.?\d+([eE][-+]?\d+)?)_db(?<db>[-+]?\d*\.?\d+([eE][-+]?\d+)?)\.mat$';
all_cfs_raw = [];
all_freqs_raw = [];
all_dbs_raw = [];

for i = 1:length(files)
    matches = regexp(files(i).name, expr, 'names');
    if ~isempty(matches)
        all_cfs_raw(end+1) = str2double(matches.cf);
        all_freqs_raw(end+1) = str2double(matches.tone);
        all_dbs_raw(end+1) = str2double(matches.db);
    end
end

% Calculate unique parameter values and sort them
unique_cfs = sort(unique(all_cfs_raw));
unique_freqs = sort(unique(all_freqs_raw)); 
unique_db = sort(unique(all_dbs_raw));

fprintf('Found %d unique CFs, %d unique frequencies, %d unique dB levels\n', ...
    length(unique_cfs), length(unique_freqs), length(unique_db));

% Create the structure in the same format as BEZ model
% For each fiber type, create a num_cf x num_freqs x num_dbs cell array
lsr_all = cell(length(unique_cfs), length(unique_freqs), length(unique_db));
msr_all = cell(length(unique_cfs), length(unique_freqs), length(unique_db));
hsr_all = cell(length(unique_cfs), length(unique_freqs), length(unique_db));

% Second pass: load and organize PSTHs into appropriate cell arrays
for i = 1:length(files)
    file_path = fullfile(input_dir, files(i).name);
    matches = regexp(files(i).name, expr, 'names');
    
    if ~isempty(matches)
        cf = str2double(matches.cf);
        freq = str2double(matches.tone);
        db = str2double(matches.db);
        
        % Find indices for this file
        [~, cf_idx] = min(abs(unique_cfs - cf));
        [~, freq_idx] = min(abs(unique_freqs - freq));
        [~, db_idx] = min(abs(unique_db - db));
        
        % Load data
        data = load(file_path);
        
        if ~isfield(data, 'psth_struct')
            warning('File %s missing psth_struct field', files(i).name);
            continue;
        end
        
        % Store PSTHs in the appropriate cell arrays (to match BEZ structure)
        if isfield(data.psth_struct, 'lsr') && ~isempty(data.psth_struct.lsr)
            lsr_all{cf_idx, freq_idx, db_idx} = data.psth_struct.lsr;
        end
        
        if isfield(data.psth_struct, 'msr') && ~isempty(data.psth_struct.msr)
            msr_all{cf_idx, freq_idx, db_idx} = data.psth_struct.msr;
        end
        
        if isfield(data.psth_struct, 'hsr') && ~isempty(data.psth_struct.hsr)
            hsr_all{cf_idx, freq_idx, db_idx} = data.psth_struct.hsr;
        end
        
        if mod(i, 50) == 0 || i == length(files)
            fprintf('Processed %d/%d files\n', i, length(files));
        end
    end
end

% Convert CFs and frequencies to strings to match BEZ format
cfs = cellfun(@num2str, num2cell(unique_cfs), 'UniformOutput', false);
frequencies = cellfun(@num2str, num2cell(unique_freqs), 'UniformOutput', false);
dbs = unique_db;

% Add a mock n_runs variable (always 1 for cochlea model)
n_runs = 1;
target_fs = 1000;

% Count filled entries for each fiber type
lsr_filled = sum(~cellfun(@isempty, lsr_all(:)));
msr_filled = sum(~cellfun(@isempty, msr_all(:)));
hsr_filled = sum(~cellfun(@isempty, hsr_all(:)));
total_cells = numel(lsr_all);

fprintf('LSR: %d/%d cells filled (%.1f%%)\n', lsr_filled, total_cells, 100*lsr_filled/total_cells);
fprintf('MSR: %d/%d cells filled (%.1f%%)\n', msr_filled, total_cells, 100*msr_filled/total_cells);
fprintf('HSR: %d/%d cells filled (%.1f%%)\n', hsr_filled, total_cells, 100*hsr_filled/total_cells);

% Check PSTH dimensions for consistency
psth_sizes = cell(3,1);
if lsr_filled > 0
    non_empty = lsr_all(~cellfun(@isempty, lsr_all));
    
    % Fix: Set UniformOutput to false
    size_strs = cellfun(@(x) sprintf('%dx%d', size(x,1), size(x,2)), non_empty, 'UniformOutput', false);
    psth_sizes{1} = unique(size_strs);
    fprintf('LSR PSTH dimensions: %s\n', strjoin(psth_sizes{1}, ', '));
end

if msr_filled > 0
    non_empty = msr_all(~cellfun(@isempty, msr_all));
    
    % Fix: Set UniformOutput to false
    size_strs = cellfun(@(x) sprintf('%dx%d', size(x,1), size(x,2)), non_empty, 'UniformOutput', false);
    psth_sizes{2} = unique(size_strs);
    fprintf('MSR PSTH dimensions: %s\n', strjoin(psth_sizes{2}, ', '));
end

if hsr_filled > 0
    non_empty = hsr_all(~cellfun(@isempty, hsr_all));
    
    % Fix: Set UniformOutput to false
    size_strs = cellfun(@(x) sprintf('%dx%d', size(x,1), size(x,2)), non_empty, 'UniformOutput', false);
    psth_sizes{3} = unique(size_strs);
    fprintf('HSR PSTH dimensions: %s\n', strjoin(psth_sizes{3}, ', '));
end

% Save the data in the same format as BEZ model
save(output_file, 'lsr_all', 'msr_all', 'hsr_all', 'cfs', 'frequencies', ...
    'dbs', 'n_runs', 'target_fs', 'fiber_types', '-v7');

fprintf('Successfully saved organized PSTH data to: %s\n', output_file);
fprintf('Data structure matches BEZ model format:\n');
fprintf('- lsr_all{cf_idx, freq_idx, db_idx}\n');
fprintf('- msr_all{cf_idx, freq_idx, db_idx}\n');
fprintf('- hsr_all{cf_idx, freq_idx, db_idx}\n');