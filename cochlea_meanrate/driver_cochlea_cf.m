% Simplified PSTH extraction
% One run only. Files are separated by (tone frequency x dB) condition.
% Produces one output MAT per (CF, tone, dB).
clear; clc;

base_dir  = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate";
input_dir = fullfile(base_dir, "out");
out_dir   = fullfile(input_dir, "condition_psths");
if ~exist(out_dir,'dir'), mkdir(out_dir); end

targetFs  = 1000;     % PSTH sampling rate (Hz)
defaultFs = 100000;   % Fallback spike time sampling

% Select tone frequencies to process (leave [] to auto-detect)
tone_freq_list = [125];   % e.g. [] or [125 250 500]

% Optional CF subset (leave [] to include all CFs found)
cf_subset = [];

if exist('compute_psth_resampled','file')~=2
    error('compute_psth_resampled.m not on MATLAB path.');
end

% Discover tones if needed
if isempty(tone_freq_list)
    tone_freq_list = discover_tones(input_dir);
end
fprintf('Tones: %s\n', mat2str(tone_freq_list));

file_list = dir(fullfile(input_dir,'*.mat'));
if isempty(file_list)
    error('No .mat files found in %s', input_dir);
end

for it = 1:numel(tone_freq_list)
    tone_val = tone_freq_list(it);
    fprintf('\n=== Tone %g (%d/%d) ===\n', tone_val, it, numel(tone_freq_list));

    % Gather files for this tone
    tone_files = {};
    db_values  = [];
    for k = 1:numel(file_list)
        fpath = fullfile(file_list(k).folder, file_list(k).name);
        try
            Smeta = load(fpath,'tone_freq','db');
        catch
            continue;
        end
        if ~isfield(Smeta,'tone_freq') || ~isfield(Smeta,'db'), continue; end
        tf = Smeta.tone_freq(:);
        if any(abs(tf - tone_val) < 1e-12)
            if isscalar(Smeta.db), db_scalar = Smeta.db; else, db_scalar = Smeta.db(1); end
            tone_files{end+1} = fpath; %#ok<AGROW>
            db_values(end+1)   = db_scalar; %#ok<AGROW>
        end
    end

    if isempty(tone_files)
        fprintf('  No files for tone %g. Skip.\n', tone_val);
        continue;
    end

    db_levels = unique(db_values);
    fprintf('  Found %d dB levels.\n', numel(db_levels));

    % Determine CF list
    if isempty(cf_subset)
        cf_all = [];
        for k = 1:numel(tone_files)
            try
                Scf = load(tone_files{k},'cf');
            catch
                continue;
            end
            if isfield(Scf,'cf') && ~isempty(Scf.cf)
                cf_all = [cf_all; Scf.cf(:)]; %#ok<AGROW>
            end
        end
        cf_list = unique(cf_all);
    else
        cf_list = cf_subset(:);
    end
    fprintf('  CF count: %d\n', numel(cf_list));

    for icf = 1:numel(cf_list)
        cf_val = cf_list(icf);
        fprintf('    CF %g (%d/%d)\n', cf_val, icf, numel(cf_list));

        for idb = 1:numel(db_levels)
            db_val = db_levels(idb);

            % Match file for this (tone, dB)
            fmatch = '';
            for k = 1:numel(tone_files)
                if db_values(k) == db_val
                    fmatch = tone_files{k}; break;
                end
            end
            if isempty(fmatch)
                fprintf('      dB=%g file missing -> skip\n', db_val);
                continue;
            end

            S = load(fmatch,'cf','spikes','anf_type','fs','tone_freq','db');
            if ~isfield(S,'cf') || ~isfield(S,'spikes')
                fprintf('      dB=%g missing cf/spikes -> skip\n', db_val);
                continue;
            end

            cf_vec = S.cf(:);
            idx_cf = find(abs(cf_vec - cf_val) < 1e-12, 1);
            if isempty(idx_cf)
                fprintf('      dB=%g CF not present -> skip\n', db_val);
                continue;
            end

            % Tone & dB validation if variable per CF
            if isfield(S,'tone_freq')
                tf_vec = S.tone_freq(:);
                tone_here = tf_vec(min(idx_cf,numel(tf_vec)));
                if abs(tone_here - tone_val) > 1e-12
                    fprintf('      dB=%g tone mismatch -> skip\n', db_val);
                    continue;
                end
            end
            if isfield(S,'db')
                db_vec_full = S.db(:);
                db_here = db_vec_full(min(idx_cf,numel(db_vec_full)));
                if abs(db_here - db_val) > 1e-12
                    fprintf('      dB=%g level mismatch -> skip\n', db_val);
                    continue;
                end
            end

            % ------ Fiber extraction for flat layout ------
            % spikes: (nCF * fibersPerCF) x 1 cell
            if ~iscell(S.spikes)
                fprintf('      dB=%g spikes not cell -> skip\n', db_val);
                continue;
            end
            nCF = numel(cf_vec);
            totalFibers = numel(S.spikes);
            if mod(totalFibers, nCF) ~= 0
                fprintf('      dB=%g inconsistent spikes length -> skip\n', db_val);
                continue;
            end
            fibersPerCF = totalFibers / nCF;

            % ------ Fiber extraction for flat layout ------
% spikes: (nCF * fibersPerCF) x 1 cell
if ~iscell(S.spikes)
    fprintf('      dB=%g spikes not cell -> skip\n', db_val);
    continue;
end

cf_vec = S.cf(:);
idx_cf_all = find(abs(cf_vec - cf_val) < 1e-12);
if isempty(idx_cf_all)
    fprintf('      dB=%g CF not present -> skip\n', db_val);
    continue;
end

segSpikes = S.spikes(idx_cf_all);

% Derive labels from anf_type
labelsSegment = {};
if isfield(S,'anf_type') && ~isempty(S.anf_type)
    if ischar(S.anf_type) && size(S.anf_type,1) == numel(cf_vec) * (numel(S.spikes)/numel(cf_vec))
        % anf_type is a char matrix: slice rows properly
        allLabels = cellstr(S.anf_type(idx_cf_all, :)); 
    elseif iscell(S.anf_type) && numel(S.anf_type) == numel(S.spikes)
        % anf_type is already a cell array
        allLabels = S.anf_type(idx_cf_all);
    else
        fprintf('      dB=%g invalid anf_type size -> skip CF\n', db_val);
        continue;
    end

    % Normalize strings
    labelsSegment = cellfun(@(c) lower(strtrim(char(c))), allLabels, 'uni', false);
else
    fprintf('      dB=%g no valid anf_type labels -> skip CF\n', db_val);
    continue;
end

% Group indices by type
idx_lsr = find(strcmp(labelsSegment,'lsr'));
idx_msr = find(strcmp(labelsSegment,'msr'));
idx_hsr = find(strcmp(labelsSegment,'hsr'));

spk_by_type = struct( ...
    'lsr', {segSpikes(idx_lsr)}, ...
    'msr', {segSpikes(idx_msr)}, ...
    'hsr', {segSpikes(idx_hsr)} );

if isempty(spk_by_type.lsr) && isempty(spk_by_type.msr) && isempty(spk_by_type.hsr)
    fprintf('      dB=%g no recognized fiber types -> skip\n', db_val);
    continue;
end

% fs handling (works regardless of ordering)
if isfield(S,'fs') && ~isempty(S.fs)
    fs_vec = S.fs(:);
    if isscalar(S.fs)
        fs_seg = repmat(S.fs, numel(idx_cf_all), 1);
    elseif numel(fs_vec) == numel(cf_vec)
        fs_seg = repmat(fs_vec(idx_cf_all(1)), numel(idx_cf_all), 1);
    elseif numel(fs_vec) == numel(S.spikes)
        fs_seg = fs_vec(idx_cf_all);
    else
        fs_seg = repmat(fs_vec(1), numel(idx_cf_all), 1);
    end
else
    fs_seg = repmat(defaultFs, numel(idx_cf_all), 1);
end

            % Compute PSTHs per type
            fiber_types = ["lsr","msr","hsr"];
            psth_struct = struct('lsr',[],'msr',[],'hsr',[]);
            type_lengths = [];

            for ft = fiber_types
                fibCells = spk_by_type.(ft);
                if isempty(fibCells), continue; end
                psths = cell(numel(fibCells),1);
                Lmin = inf;
                for f = 1:numel(fibCells)
                    spikes_f = fibCells{f};
                    if isempty(spikes_f), continue; end
                    psths{f} = compute_psth_resampled(spikes_f, fs_seg(1), targetFs);
                    Lmin = min(Lmin, numel(psths{f}));
                end
                if isinf(Lmin)
                    continue;
                end
                for f = 1:numel(psths)
                    if numel(psths{f}) > Lmin
                        psths{f} = psths{f}(1:Lmin);
                    end
                end
                M = cell2mat(cellfun(@(v)v(:), psths,'uni',false));
                psth_struct.(ft) = mean(M,2);
                type_lengths(end+1) = numel(psth_struct.(ft)); %#ok<AGROW>
            end

            if isempty(type_lengths)
                fprintf('      dB=%g no PSTHs -> skip\n', db_val);
                continue;
            end

            % Harmonize length across types (min)
            Lfinal = min(type_lengths);
            if ~isempty(psth_struct.lsr), psth_struct.lsr = psth_struct.lsr(1:Lfinal); end
            if ~isempty(psth_struct.msr), psth_struct.msr = psth_struct.msr(1:Lfinal); end
            if ~isempty(psth_struct.hsr), psth_struct.hsr = psth_struct.hsr(1:Lfinal); end

            time = (0:Lfinal-1)'/targetFs;

            psth_lsr = []; mean_rate_lsr = [];
            psth_msr = []; mean_rate_msr = [];
            psth_hsr = []; mean_rate_hsr = [];
            if ~isempty(psth_struct.lsr), psth_lsr = psth_struct.lsr; mean_rate_lsr = mean(psth_lsr); end
            if ~isempty(psth_struct.msr), psth_msr = psth_struct.msr; mean_rate_msr = mean(psth_msr); end
            if ~isempty(psth_struct.hsr), psth_hsr = psth_struct.hsr; mean_rate_hsr = mean(psth_hsr); end

            save_name = sprintf('psth_cf%g_tone%g_db%g.mat', cf_val, tone_val, db_val);
            cf_out = cf_val; tone_out = tone_val; db_out = db_val; run_id = 0; %#ok<NASGU>
            save(fullfile(out_dir, save_name), 'time','targetFs', ...
                'psth_lsr','psth_msr','psth_hsr', ...
                'mean_rate_lsr','mean_rate_msr','mean_rate_hsr', ...
                'cf_out','tone_out','db_out','run_id');
            fprintf('      Saved %s (lsr=%d msr=%d hsr=%d)\n', save_name, ...
                ~isempty(psth_lsr), ~isempty(psth_msr), ~isempty(psth_hsr));

            clear psth_struct spk_by_type psth_lsr psth_msr psth_hsr mean_rate_lsr mean_rate_msr mean_rate_hsr time;
        end
    end
end

fprintf('\nDone.\n');

% -------- Helper to auto-discover tones --------
function tones = discover_tones(input_dir)
files = dir(fullfile(input_dir,'*.mat'));
tones_accum = [];
for k = 1:numel(files)
    try
        S = load(fullfile(files(k).folder, files(k).name),'tone_freq');
    catch
        continue;
    end
    if isfield(S,'tone_freq') && ~isempty(S.tone_freq)
        tones_accum = [tones_accum; S.tone_freq(:)]; %#ok<AGROW>
    end
end
tones = unique(tones_accum);
end

