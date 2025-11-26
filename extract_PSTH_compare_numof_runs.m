folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF";
fiber_str = '32-32-32';
run_types = {'2runs', '4runs', '8runs', '16runs'};  % group by these
num_cf = 3;
num_tone = 5;
num_db = 4;
num_pairs = num_cf * num_tone * num_db;

for rt = 1:length(run_types)
    run_tag = run_types{rt};
    fprintf('\nProcessing run type: %s\n', run_tag);

    % Find matching files
    file_pattern = sprintf('synapse_results_%s_run*_*.mat', run_tag);
    file_list = dir(fullfile(folder, file_pattern));

    if isempty(file_list)
        warning('No files found for %s', run_tag);
        continue;
    end

    % Initialize storage
    psth_runs_lsr = cell(num_pairs, length(file_list));
    psth_runs_msr = cell(num_pairs, length(file_list));
    psth_runs_hsr = cell(num_pairs, length(file_list));

    % Process each file
    for f = 1:length(file_list)
        file = file_list(f).name;
        fprintf(' Loading file: %s\n', file);
        data = load(fullfile(folder, file));
        results = data.all_synapse_results;

        for i_pair = 1:num_pairs
            syn_df = results{1,i_pair};

            % LSR
            lsr_idx = strcmp(syn_df.anf_type, 'lsr');
            lsr_mat = cell2mat(syn_df.psth(lsr_idx)); % [32, T]
            psth_runs_lsr{i_pair, f} = mean(lsr_mat);

            % MSR
            msr_idx = strcmp(syn_df.anf_type, 'msr');
            msr_mat = cell2mat(syn_df.psth(msr_idx));
            psth_runs_msr{i_pair, f} = mean(msr_mat);

            % HSR
            hsr_idx = strcmp(syn_df.anf_type, 'hsr');
            hsr_mat = cell2mat(syn_df.psth(hsr_idx));
            psth_runs_hsr{i_pair, f} = mean(hsr_mat);
        end
    end

    % Load metadata from any file
    meta = load(fullfile(folder, file_list(1).name));
    meta = meta.all_synapse_results;

    % Prepare output
    cf_value = zeros(num_pairs, 1);
    cf_index = zeros(num_pairs, 1);
    freq_value = zeros(num_pairs, 1);
    db_value = zeros(num_pairs, 1);
    tone_index = zeros(num_pairs, 1);

    mean_lsr = cell(num_pairs, 1);
    mean_msr = cell(num_pairs, 1);
    mean_hsr = cell(num_pairs, 1);

    psth_all_runs_lsr = cell(num_pairs, 1);
    psth_all_runs_msr = cell(num_pairs, 1);
    psth_all_runs_hsr = cell(num_pairs, 1);

    for i_pair = 1:num_pairs
        syn = meta{1,i_pair};
        cf_index(i_pair) = syn.cf_idx;
        cf_value(i_pair) = syn.input_cf;
        freq_value(i_pair) = syn.input_freq;
        db_value(i_pair) = syn.input_db;
        tone_index(i_pair) = syn.input_toneidx;

        lsr_stack = cat(1, psth_runs_lsr{i_pair, :});
        msr_stack = cat(1, psth_runs_msr{i_pair, :});
        hsr_stack = cat(1, psth_runs_hsr{i_pair, :});

        psth_all_runs_lsr{i_pair} = lsr_stack;
        psth_all_runs_msr{i_pair} = msr_stack;
        psth_all_runs_hsr{i_pair} = hsr_stack;

        mean_lsr{i_pair} = mean(lsr_stack, 1);
        mean_msr{i_pair} = mean(msr_stack, 1);
        mean_hsr{i_pair} = mean(hsr_stack, 1);
    end

    % Create and save output table
    T = table(cf_index, cf_value, tone_index, freq_value, db_value, ...
        mean_lsr, mean_msr, mean_hsr, ...
        psth_all_runs_lsr, psth_all_runs_msr, psth_all_runs_hsr, ...
        'VariableNames', {'CF index', 'CF value', 'Tone Index', ...
        'Freq Value', 'Db Level', ...
        'Mean PSTH - LSR', 'Mean PSTH - MSR', 'Mean PSTH - HSR', ...
        'PSTH Runs - LSR', 'PSTH Runs - MSR', 'PSTH Runs - HSR'});

    save_name = sprintf('cf_tone_pair_across_simulations_psth_table_%s_%s.mat', run_tag, fiber_str);
    save(fullfile(folder, save_name), 'T');
    fprintf('Saved: %s\n', save_name);
end
