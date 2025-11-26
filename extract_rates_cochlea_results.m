folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate";
nruns = 8;
%fiber_groups = {'2_2_2', '4_4_4', '8_8_8', '16_16_16', ...
%    '32_32_32', '64_64_64', '128_128_128'};

fiber_groups = {'16_16_16'};


num_cf = 3;
num_tone = 5;
num_db = 4;
num_pairs = num_cf * num_tone * num_db;
for fg = 1:length(fiber_groups)
    fiber_str = fiber_groups{fg};
    fprintf('Processing fiber group: %s\n', fiber_str);

    psth_runs_lsr = containers.Map('KeyType', 'char', 'ValueType', 'any');
    psth_runs_msr = containers.Map('KeyType', 'char', 'ValueType', 'any');
    psth_runs_hsr = containers.Map('KeyType', 'char', 'ValueType', 'any');

    all_keys = {}; % Store keys as cell array 
 
    for run_i = 0:nruns-1
        filename = sprintf('cochlea_spike_rates_numcf_3_numtonefreq_5_hml_%s_run%d_nonacc.mat', ...
            fiber_str, run_i);
        filepath = fullfile(folder, filename);

        if ~isfile(filepath)
            warning('File not found: %s', filename);
            continue;
        end

        fprintf('  Loading %s\n', filename);
        D = load(filepath);

        % Extract fields
        anf_type = strtrim(cellstr(D.anf_type));  % 360x1 cell array
        cf = D.cf;
        db = D.db;
        tone_freq = D.tone_freq;
        spikes = D.spikes;  % 360x1 cell, each 20,000x1 single

        % Identify unique stimulus combinations
        stim_matrix = [cf, tone_freq, db];
        [unique_rows, ~, group_ids] = unique(stim_matrix, 'rows');

        for i = 1:size(unique_rows, 1)
            key = sprintf('cf%.1f_freq%.1f_db%.1f', unique_rows(i,1), unique_rows(i,2), unique_rows(i,3));
            idxs = find(group_ids == i);

            % LSR
            lsr_i = idxs(strcmp(anf_type(idxs), 'lsr'));
            if ~isempty(lsr_i)
                lsr_cells = spikes(lsr_i);
                lsr_rows = cellfun(@(x) x(:)', lsr_cells, 'UniformOutput', false);
                lsr_mat = cell2mat(lsr_rows); % now [num_lsr_fibers x 200]
                psth_lsr = mean(lsr_mat, 1);

                if ~isKey(psth_runs_lsr, key)
                    psth_runs_lsr(key) = {{}};
                end
                curr_cell = psth_runs_lsr(key);
                curr_array = curr_cell{1};
                curr_array{end+1} = psth_lsr;
                psth_runs_lsr(key) = {curr_array};
            end

            % MSR
            msr_i = idxs(strcmp(anf_type(idxs), 'msr'));
            if ~isempty(msr_i)
                msr_cells = spikes(msr_i);
                msr_rows = cellfun(@(x) x(:)', msr_cells, 'UniformOutput', false);
                msr_mat = cell2mat(msr_rows);
                psth_msr = mean(msr_mat, 1);

                if ~isKey(psth_runs_msr, key)
                    psth_runs_msr(key) = {{}};
                end
                curr_cell = psth_runs_msr(key);
                curr_array = curr_cell{1};
                curr_array{end+1} = psth_msr;
                psth_runs_msr(key) = {curr_array};
            end

            % HSR
            hsr_i = idxs(strcmp(anf_type(idxs), 'hsr'));
            if ~isempty(hsr_i)
                hsr_cells = spikes(hsr_i);
                hsr_rows = cellfun(@(x) x(:)', hsr_cells, 'UniformOutput', false);
                hsr_mat = cell2mat(hsr_rows);
                psth_hsr = mean(hsr_mat, 1);

                if ~isKey(psth_runs_hsr, key)
                    psth_runs_hsr(key) = {{}};
                end
                curr_cell = psth_runs_hsr(key);
                curr_array = curr_cell{1};
                curr_array{end+1} = psth_hsr;
                psth_runs_hsr(key) = {curr_array};
            end

            all_keys{end+1} = key;
        end
    end

    % Deduplicate stimulus keys
    all_keys = unique(all_keys);

    % Construct final table
    n_pairs = length(all_keys);
    cf_value = zeros(n_pairs,1);
    freq_value = zeros(n_pairs,1);
    db_value = zeros(n_pairs,1);

    mean_lsr = cell(n_pairs,1);
    mean_msr = cell(n_pairs,1);
    mean_hsr = cell(n_pairs,1);

    psth_all_runs_lsr = cell(n_pairs,1);
    psth_all_runs_msr = cell(n_pairs,1);
    psth_all_runs_hsr = cell(n_pairs,1);

    for i = 1:n_pairs
        key = all_keys{i};
        tokens = sscanf(key, 'cf%f_freq%f_db%f');
        cf_value(i) = tokens(1);
        freq_value(i) = tokens(2);
        db_value(i) = tokens(3);

        % LSR
        if isKey(psth_runs_lsr, key)
            runs_cell = psth_runs_lsr(key);  % This is a 1x1 cell golding the actual cell array
            runs = runs_cell{1};                % Unfwap one layer to get cell array
            psth_all_runs_lsr{i} = cat(1, runs{:}); % 8x200
            mean_rate_per_run = mean(psth_all_runs_lsr{i},2); % Mean over time bins, per run, 8x1
            mean_lsr{i} = mean(mean_rate_per_run);
        end

        % MSR
        if isKey(psth_runs_msr, key)
            runs_cell = psth_runs_msr(key);
            runs = runs_cell{1};
            psth_all_runs_msr{i} = cat(1, runs{:});
            mean_rate_per_run = mean(psth_all_runs_msr{i}, 2);
            mean_msr{i} = mean(mean_rate_per_run);
        end

        % HSR
        if isKey(psth_runs_hsr, key)
            runs_cell = psth_runs_hsr(key);
            runs = runs_cell{1};
            psth_all_runs_hsr{i} = cat(1, runs{:});
            mean_rate_per_run = mean(psth_all_runs_hsr{i}, 2);
            mean_hsr{i} = mean(mean_rate_per_run);
        end
    end

    T = table(cf_value, freq_value, db_value, ...
        mean_lsr, mean_msr, mean_hsr, ...
        psth_all_runs_lsr, psth_all_runs_msr, psth_all_runs_hsr, ...
        'VariableNames', {'CF_Hz', 'ToneFreq_Hz', 'DB_SPL', ...
        'MeanPSTH_LSR', 'MeanPSTH_MSR', 'MeanPSTH_HSR', ...
        'PSTHs_LSR', 'PSTHs_MSR', 'PSTHs_HSR'});
    T = sortrows(T, {'CF_Hz', 'ToneFreq_Hz', 'DB_SPL'})
    % Save
    out_name = sprintf('cf_tone_pair_across_runs_psth_table_%s.mat', fiber_str);
    save(fullfile(folder, out_name), 'T');
    fprintf('Saved: %s\n', out_name);
end