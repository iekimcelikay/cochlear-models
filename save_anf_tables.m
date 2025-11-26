num_cf = 3;
num_tone_freq = 5;
num_db_level = 4;
num_lsr = 512;
num_msr = 512;
num_hsr = 512;

load("synapse_results_20250701_165154_512-512-512 (copy).mat");

% As I resample the PSTH, 


number_of_cf_tone_pairs = num_cf * num_tone_freq * num_db_level;
disp(numel(number_of_cf_tone_pairs))
lsr_all_fibers_psth_per_tone_cf = [];
cf_value = [];
cf_index = [];
freq_value = [];
db_value = [];
tone_index = [];
cf_tone_mean_psths = struct();

for i_pair=1:number_of_cf_tone_pairs
    % This is the synapse df for each tone-cf pair. (i.e: i_pair =1, means
    % 1 of 60 cf-tone pairs) 
    synapse_df_pair = all_synapse_results{1,i_pair};
    % Extract information
    cf_value(i_pair,:) = synapse_df_pair.input_cf;
    cf_index(i_pair,:) = synapse_df_pair.cf_idx; % This originally starts from 0, 
    freq_value(i_pair,:) = synapse_df_pair.input_freq;
    db_value(i_pair,:) = synapse_df_pair.input_db;
    tone_index(i_pair,:) = synapse_df_pair.input_toneidx;

    % FOR LSR 
    lsr_indices = find(strcmp(synapse_df_pair.anf_type, 'lsr'));
    lsr_psth_tone_cf_pair = synapse_df_pair.psth(lsr_indices);

    for i_lsr = 1:num_lsr
    lsr_fiber_psth = lsr_psth_tone_cf_pair{1,i_lsr};
    lsr_all_fibers_psth_per_tone_cf(i_lsr,:) = lsr_fiber_psth;
    % Then I want to average over the number of fibers in this category
    lsr_mean_all_psth_per_tone_cf = mean(lsr_all_fibers_psth_per_tone_cf);
    end

    cf_tone_mean_psths.lsr{i_pair,:} = lsr_mean_all_psth_per_tone_cf;


    % FOR MSR
    msr_indices = find(strcmp(synapse_df_pair.anf_type, 'msr'));
    msr_psth_tone_cf_pair = synapse_df_pair.psth(msr_indices);

    for i_msr = 1:num_msr
    msr_fiber_psth = msr_psth_tone_cf_pair{1,i_msr};
    msr_all_fibers_psth_per_tone_cf(i_msr,:) = msr_fiber_psth;
    % Then I want to average over the number of fibers in this category
    msr_mean_all_psth_per_tone_cf = mean(msr_all_fibers_psth_per_tone_cf);
    end

    cf_tone_mean_psths.msr{i_pair,:} = msr_mean_all_psth_per_tone_cf;

     % FOR HSR
    hsr_indices = find(strcmp(synapse_df_pair.anf_type, 'hsr'));
    hsr_psth_tone_cf_pair = synapse_df_pair.psth(hsr_indices);

    for i_hsr = 1:num_hsr   
    hsr_fiber_psth = hsr_psth_tone_cf_pair{1,i_hsr};
    hsr_all_fibers_psth_per_tone_cf(i_hsr,:) = hsr_fiber_psth;
    % Then I want to average over the number of fibers in this category
    hsr_mean_all_psth_per_tone_cf = mean(hsr_all_fibers_psth_per_tone_cf);
    end

    cf_tone_mean_psths.hsr{i_pair,:} = hsr_mean_all_psth_per_tone_cf;

end

cf_tone_mean_psths.num_cf = num_cf;
cf_tone_mean_psths.num_tone_freq = num_tone_freq; 
cf_tone_mean_psths.num_db_lvl = num_db_level;

cf_tone_pair_data = table(cf_index, cf_value, tone_index, freq_value, ...
    db_value, cf_tone_mean_psths.lsr, cf_tone_mean_psths.msr, cf_tone_mean_psths.hsr, ...
    'VariableNames', ["CF index", "CF value", "Tone Index", "Freq Value", "Db Level", ...
    "Mean PSTH - 512 LSR", "Mean PSTH - 512 MSR", "Mean PSTH - 512 HSR"]);
save("cf_tone_pair_table_3x5x4_512-512-512.mat", "cf_tone_pair_data");

