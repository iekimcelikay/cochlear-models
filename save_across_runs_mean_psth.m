
folder ="/home/ekim/PycharmProjects/subcorticalSTRF/results";

nruns = 10;
fiber_groups = {'2-2-2', '4-4-4', '8-8-8', '16-16-16', '32-32-32', '64-64-64', '128-128-128', '256-256-256', '512-512-512'};

num_cf = 3;
num_tone = 5;
num_db = 4;
num_pairs = num_cf * num_tone * num_db; 

for fg = 1:length(fiber_groups)
    fiber_str = fiber_groups{fg};
    fprintf('Processing fiber group: %s\n', fiber_str);

    % Storage for PSTHs across runs 
    psth_runs_lsr = cell(num_pairs, nruns);
    psth_runs_msr = cell(num_pairs, nruns);
    psth_runs_hsr = cell(num_pairs, nruns);


    for run_i = 0:nruns-1
        filename = sprintf('synapse_results_run%d_%s_RandomNoise.mat', run_i, fiber_str);
        filepath = fullfile(folder, filename);

        if exist(filepath, 'file')
            fprintf(' Loading %s\n', filename);
            data = load(filepath);
            results = data.all_synapse_results;

            for i_pair = 1:num_pairs
                syn_df = results{1,i_pair};

                % LSR
                lsr_idx = strcmp(syn_df.anf_type, 'lsr');
                lsr_mat = cell2mat(syn_df.psth(lsr_idx)); % [N_lsr, T]
                psth_runs_lsr{i_pair, run_i+1} = mean(lsr_mat);

                % MSR
                msr_idx = strcmp(syn_df.anf_type, 'msr');
                msr_mat = cell2mat(syn_df.psth(msr_idx)); % [N_msr, T]
                psth_runs_msr{i_pair, run_i+1} = mean(msr_mat);
                
                % HSR
                hsr_idx = strcmp(syn_df.anf_type, 'hsr');
                hsr_mat = cell2mat(syn_df.psth(hsr_idx)); % [N_hsr, T]
                psth_runs_hsr{i_pair, run_i+1} = mean(hsr_mat);
            end
        else
            warning('File not found: %s', filename);
        end
    end

    % Initialize output structures
    mean_lsr = cell(num_pairs,1);
    mean_msr = cell(num_pairs,1);
    mean_hsr = cell(num_pairs,1);

    psth_all_runs_lsr = cell(num_pairs, 1);
    psth_all_runs_msr = cell(num_pairs,1);
    psth_all_runs_hsr = cell(num_pairs,1);

    % Load meta info from first valid run 
    meta_loaded = false;
    for r = 1:nruns
        if ~isempty(psth_runs_lsr{1,r})
            sample_data = load(fullfile(folder, ...
                sprintf('synapse_results_run%d_%s_RandomNoise.mat', r-1, fiber_str)));
            meta = sample_data.all_synapse_results;
            meta_loaded = true;
            break;
        end
    end
    if ~meta_loaded
        warning('No valid runs for fiber group %s', fiber_str);
        continue;
    end

    cf_value = zeros(num_pairs, 1);
    cf_index = zeros(num_pairs, 1);
    freq_value = zeros(num_pairs,1);
    db_value = zeros(num_pairs,1);
    tone_index = zeros(num_pairs,1);

    for i_pair = 1:num_pairs 
        % Extract metadata from one run 
        syn = meta{1,i_pair};
        cf_index(i_pair) = syn.cf_idx;
        cf_value(i_pair) = syn.input_cf;
        freq_value(i_pair) = syn.input_freq;
        db_value(i_pair) = syn.input_db;
        tone_index(i_pair) = syn.input_toneidx;

        % Stack PSTHs across runs 
        lsr_stack = cat(1, psth_runs_lsr{i_pair, :});
        msr_stack = cat(1, psth_runs_msr{i_pair, :});
        hsr_stack = cat(1, psth_runs_hsr{i_pair, :});

        % Save raw data
        psth_all_runs_lsr{i_pair} = lsr_stack;
        psth_all_runs_msr{i_pair} = msr_stack;
        psth_all_runs_hsr{i_pair} = hsr_stack;
        
        % Compute mean across runs 
        mean_lsr{i_pair} = mean(lsr_stack, 1);

        mean_msr{i_pair} = mean(msr_stack, 1);

        mean_hsr{i_pair} = mean(hsr_stack, 1);

    end

    % Create output table
    T= table(cf_index, cf_value, tone_index, freq_value, db_value, ...
        mean_lsr, mean_msr, mean_hsr, ...
        psth_all_runs_lsr, psth_all_runs_msr, psth_all_runs_hsr,...
        'VariableNames', {'CF index', 'CF value', 'Tone Index', ...
        'Freq Value', 'Db Level', ...
        'Mean PSTH - LSR',...
        'Mean PSTH - MSR',...
        'Mean PSTH - HSR',...
        'PSTH Runs - LSR', 'PSTH Runs - MSR', 'PSTH Runs - HSR'});

    % Save 
    save_filename = sprintf('cf_tone_pair_across_runs_psth_table_%s.mat', fiber_str);
    save(fullfile(folder, save_filename), 'T');
    fprintf('Saved result: %s\n', save_filename);
end

