
folder ="/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/results";
n_runs = 8;
num_cf = 5;
cfs = {'125.0', '332.9', '700.5', '1350.5', '2500.0'};
frequencies = {'125.0', '332.9', '700.5','1350.5', '2500.0'};
dbs = [50, 60, 70, 80];
fs = 100000;
target_fs = 1000;


fiber_groups = {'2-2-2', '4-4-4', '8-8-8', '16-16-16', '32-32-32', '64-64-64', '128-128-128', '256-256-256', '512-512-512'};
num_pairs = num_cf * length(frequencies)* length(dbs);

num_fiber_groups = length(fiber_groups);

% Preallocate the struct array
all_psth_data = repmat(struct('lsr', [], 'msr', [], 'hsr', []), 1, num_fiber_groups);
% Initialize an empty table to store the results
% Each row will correspond to a specific combination of CF, tone, db, and run
results_table = table();
for i_fibergroup = 1:num_fiber_groups

    all_psth_data(i_fibergroup).lsr = cell(num_cf, length(frequencies), length(dbs), n_runs);
    all_psth_data(i_fibergroup).msr = cell(num_cf, length(frequencies), length(dbs), n_runs);
    all_psth_data(i_fibergroup).hsr = cell(num_cf, length(frequencies), length(dbs), n_runs);
	fiber_str = fiber_groups{i_fibergroup};
	fprintf('Processing fiber group: %s\n', fiber_str);

	folder_name = sprintf('psth_batch_%druns_%dcfs_%sfibers', n_runs, num_cf, fiber_str);
	folder_path = fullfile(folder, folder_name);

	for run_i = 0:n_runs-1
		for i_cf = 1:length(cfs)
			cf_val = string(cfs{i_cf});
			for i_tonefreq = 1:length(frequencies)
				tonefreq_val = string(frequencies{i_tonefreq});
				for i_db = 1:length(dbs)
					db_val = dbs(i_db);
					filename = sprintf('synapse_result_cf%shz_tone%shz_%ddb_%sfibers_run%d.mat', cf_val, tonefreq_val, db_val, fiber_str, run_i);

					filepath = fullfile(folder_path, filename);


						fprintf(' Loading %s\n', filename);
						data = load(filepath);
						results = data.synapse_result_struct;
						no_fibers = length(results.psth);

						% LSR
                         
						lsr_idx = strcmp(results.anf_type, 'lsr');
                        lsr_spikes = results.psth(lsr_idx);

                        % Compute PSTH for each LSR fiber
                        lsr_psths = cellfun(@(spk) compute_psth_resampled(spk, fs, target_fs), lsr_spikes, 'UniformOutput', false);

                        %Convert to matrix: each column is one fiber's PSTH
                        lsr_mat = cell2mat(cellfun(@(x) x(:), lsr_psths, 'UniformOutput', false));

                        % Average across fibers
                        lsr_mean_psth = mean(lsr_mat,2);
                        all_psth_data(i_fibergroup).lsr{i_cf, i_tonefreq, i_db, run_i+1} = lsr_mean_psth;
                
                         % MSR
                        msr_idx = strcmp(results.anf_type, 'msr');
                        msr_spikes = results.psth(msr_idx);
                        msr_psths = cellfun(@(spk) compute_psth_resampled(spk, fs, target_fs), ...
                                    msr_spikes, 'UniformOutput', false);
                        msr_mat = cell2mat(cellfun(@(x) x(:), msr_psths, 'UniformOutput', false));
                        msr_mean_psth = mean(msr_mat, 2);
                        all_psth_data(i_fibergroup).msr{i_cf, i_tonefreq, i_db, run_i+1} = msr_mean_psth;
                       
                        % HSR
                        hsr_idx = strcmp(results.anf_type, 'hsr');
                        hsr_spikes = results.psth(hsr_idx);
                        hsr_psths = cellfun(@(spk) compute_psth_resampled(spk, fs, target_fs), ...
                                    hsr_spikes, 'UniformOutput', false);
                        hsr_mat = cell2mat(cellfun(@(x) x(:), hsr_psths, 'UniformOutput', false));
                        hsr_mean_psth = mean(hsr_mat, 2);
                        all_psth_data(i_fibergroup).hsr{i_cf, i_tonefreq, i_db, run_i+1} = hsr_mean_psth;
                    
                end
            end
        end
    end
end

%% Save
save('all_psth_data.mat', 'all_psth_data', 'fiber_groups', 'cfs', 'frequencies', 'dbs');

						
%%
fiber_counts = cellfun(@(x) sscanf(x, '%d', 1), fiber_groups);


mean_rates = struct();
mean_rates.lsr = zeros(num_cf, length(frequencies), length(dbs), length(fiber_groups));
mean_rates.msr = zeros(num_cf, length(frequencies), length(dbs), length(fiber_groups));
mean_rates.hsr = zeros(num_cf, length(frequencies), length(dbs), length(fiber_groups));

for fg_i = 1:length(fiber_groups)
    for i_cf = 1:num_cf
        for i_tone = 1:length(frequencies)
            for i_db = 1:length(dbs)
                % Extract PSTHs for each run (cell array of PSTH vectors)
                lsr_runs = squeeze(all_psth_data(fg_i).lsr(i_cf, i_tone, i_db, :));
                msr_runs = squeeze(all_psth_data(fg_i).msr(i_cf, i_tone, i_db, :));
                hsr_runs = squeeze(all_psth_data(fg_i).hsr(i_cf, i_tone, i_db, :));

                % Convert cell arrays to matrices: each column = PSTH for one run
                lsr_mat = cell2mat(cellfun(@(x) x(:), lsr_runs, 'UniformOutput', false));
                msr_mat = cell2mat(cellfun(@(x) x(:), msr_runs, 'UniformOutput', false));
                hsr_mat = cell2mat(cellfun(@(x) x(:), hsr_runs, 'UniformOutput', false));

                % Average PSTH across runs (mean over columns)
                mean_lsr_psth = mean(lsr_mat, 2);
                mean_msr_psth = mean(msr_mat, 2);
                mean_hsr_psth = mean(hsr_mat, 2);

                % Mean firing rate = average over time bins
                mean_rates.lsr(i_cf, i_tone, i_db, fg_i) = mean(mean_lsr_psth);
                mean_rates.msr(i_cf, i_tone, i_db, fg_i) = mean(mean_msr_psth);
                mean_rates.hsr(i_cf, i_tone, i_db, fg_i) = mean(mean_hsr_psth);
            end
        end
    end
end
% Save to .mat for later comparison
save('mean_rates_all_fibers.mat', 'mean_rates', 'sem_rates', 'fiber_counts', 'cfs', 'frequencies', 'dbs');


%% Check nans
nan_entries = [];

for fg_i = 1:length(fiber_groups)
    for i_cf = 1:num_cf
        for i_tone = 1:length(frequencies)
            for i_db = 1:length(dbs)
                for ftype_i = 1:3
                    field = fiber_type_fields{ftype_i};
                    val = mean_rates.(field)(i_cf, i_tone, i_db, fg_i);
                    if isnan(val)
                        nan_entries = [nan_entries; fg_i, i_cf, i_tone, i_db, ftype_i];
                    end
                end
            end
        end
    end
end

% Display results
if isempty(nan_entries)
    fprintf('No NaNs found in mean_rates!\n');
else
    fprintf('NaNs detected at these indices: fiber_group, CF, tone, dB, fiber_type\n');
    disp(nan_entries);
end


%% CURRENT PLOT
fiber_counts = cellfun(@(x) sscanf(x, '%d', 1), fiber_groups);
fiber_type_names = {'LSR', 'MSR', 'HSR'};
fiber_type_fields = {'lsr', 'msr', 'hsr'};
line_styles = {'-', '--', ':'};
colors = lines(length(fiber_type_fields)); % color by fiber type
convergence_eps = 0.05; % ±5% tolerance

for i_cf = 1:num_cf
    cf_value = str2double(cfs{i_cf});
    conv_data = {};
    
    figure('Name', sprintf('CF = %g Hz', cf_value), 'NumberTitle', 'off');
    sgtitle(sprintf('Mean Firing Rate vs Fiber Group Size – CF = %g Hz', cf_value));
    
    num_freqs = length(frequencies);
    num_dbs = length(dbs);
    
    t = tiledlayout(num_dbs, num_freqs, 'Padding','compact','TileSpacing','compact');
    
    for i_db = 1:num_dbs
        for i_tone = 1:num_freqs
            tile_idx = (i_db-1)*num_freqs + i_tone; % row-major order
            ax = nexttile(tile_idx);
            hold(ax,'on');
            
            for ftype_i = 1:length(fiber_type_fields)
                field = fiber_type_fields{ftype_i};
                mean_rates_vs_n = zeros(1, length(fiber_groups));
                sem_rates_vs_n = zeros(1, length(fiber_groups));
                
                for fg_i = 1:length(fiber_groups)
                    psth_runs = squeeze(all_psth_data(fg_i).(field)(i_cf, i_tone, i_db, :));
                    
                    if isempty(psth_runs)
                        mean_rates_vs_n(fg_i) = NaN;
                        sem_rates_vs_n(fg_i) = NaN;
                        continue;
                    end
                    
                    psth_mat = cell2mat(reshape(cellfun(@(x) x(:), psth_runs, 'UniformOutput', false),1,[]));
                    psth_mat = reshape(psth_mat, length(psth_runs{1}), []);
                    
                    mean_rate_per_run = mean(psth_mat, 1);
                    
                    mean_rates_vs_n(fg_i) = mean(mean_rate_per_run);
                    sem_rates_vs_n(fg_i) = std(mean_rate_per_run)/sqrt(size(psth_mat,2));
                end
                
                % Normalize to largest fiber group
                ref_rate = mean_rates_vs_n(end);
                norm_rates = mean_rates_vs_n / ref_rate;
                % convergence
                abs_error = abs(norm_rates - 1);
                conv_idx = find(abs_error < convergence_eps, 1);
                if ~isempty(conv_idx)
                    conv_n = fiber_counts(conv_idx);
                else
                    conv_n = NaN;
                end
                norm_sem = sem_rates_vs_n / ref_rate;
                conv_data(end+1,:) = {cf_value, str2double(frequencies{i_tone}), dbs(i_db), fiber_type_names{ftype_i}, conv_n};
                % Plot shaded SEM
                x = fiber_counts;
                y = norm_rates(:)';
                e = norm_sem(:)';
                fill([x fliplr(x)], [y+e fliplr(y-e)], colors(ftype_i,:), ...
                    'FaceAlpha',0.2,'EdgeColor','none','Parent',ax);
                
                % Plot mean line
                plot(ax, x, y, 'Color', colors(ftype_i,:), 'LineStyle', line_styles{ftype_i}, ...
                    'Marker','o','LineWidth',1.5,'DisplayName',fiber_type_names{ftype_i});
            end
            
            xlabel(ax,'Number of fibers');
            ylabel(ax,'Normalized mean firing rate');
            title(ax,sprintf('Tone %s Hz – %d dB', frequencies{i_tone}, dbs(i_db)));
            set(ax,'XScale','log');
            xticks(ax,fiber_counts);
            xticklabels(ax,string(fiber_counts));
            grid(ax,'on');
            hold(ax,'off');
        end
    end

     % --- Save text file for this CF ---
% Save text file for this CF
fname = sprintf('convergence_CF_%g.txt', cf_value);
fid = fopen(fname,'w');
fprintf(fid, 'CF\tTone(Hz)\tdB\tANF_type\tConvergedFibers\n');

for i = 1:size(conv_data,1)
    CF_val       = conv_data{i,1};
    tone_val     = conv_data{i,2};
    db_val       = conv_data{i,3};
    anf_type     = conv_data{i,4};
    conv_fibers  = conv_data{i,5};
    
    % Ensure numeric fields are printed as numbers, conv_fibers can be NaN
    if isnan(conv_fibers)
        conv_str = 'NaN';
    else
        conv_str = num2str(conv_fibers);
    end
    
    fprintf(fid, '%g\t%g\t%d\t%s\t%s\n', CF_val, tone_val, db_val, anf_type, conv_str);
end

fclose(fid);
fprintf('Convergence info for CF=%g saved to %s\n', cf_value, fname);

    
    % --- Pop-up table for this CF ---
    conv_fig = figure('Name', sprintf('Convergence CF = %g Hz', cf_value), ...
                      'NumberTitle','off','Position',[100 100 600 400]);
    uitable(conv_fig, 'Data', conv_data, ...
                     'ColumnName', {'CF','Tone(Hz)','dB','ANF_type','ConvergedFibers'}, ...
                     'RowName', [], 'Units', 'Normalized', 'Position',[0 0 1 1]);
    
    % Global legend (attach to top-left subplot)
    ax_for_legend = t.Children(end); % top-left tile
    hold(ax_for_legend,'on');
    line_handles = gobjects(1,length(fiber_type_names));
    for ftype_i = 1:length(fiber_type_names)
        line_handles(ftype_i) = plot(ax_for_legend, NaN, NaN, ...
            'Color', colors(ftype_i,:), 'LineStyle', line_styles{ftype_i}, ...
            'Marker','o', 'LineWidth',1.5, ...
            'DisplayName', fiber_type_names{ftype_i});
    end
    legend(ax_for_legend, line_handles, fiber_type_names, 'Location','northeastoutside');
end


%% Retrieving the data from cell array

% Example: Retrieve LSR mean for CF = 125 Hz, Tone = 2500 Hz, db = 70, Run 1
cf_idx = 1;    % Index for CF = 125 Hz
tone_idx = 5;  % Index for tone = 2500 Hz
db_idx = 3;    % Index for db = 70
run_idx = 1;   % Index for Run 1 (since run_i starts from 0, use run_i + 1)

lsr_mean_specific = lsr_means{cf_idx, tone_idx, db_idx, run_idx};  % 20000x1 vector



%% Example: Plot average LSR for CF = 125 Hz, tone = 2500 Hz, db = 70
cf_idx = 1;    % Index for CF = 125 Hz
tone_idx = 5;  % Index for tone = 2500 Hz
db_idx = 3;    % Index for db = 70

% Extract all LSR mean vectors for this combination of CF, tone, db, across all runs
lsr_all_runs = cell2mat(lsr_means(cf_idx, tone_idx, db_idx, :));

% Compute the average LSR across runs (mean across the rows)
avg_lsr = mean(lsr_all_runs, 2);  % This will give you a 20000x1 vector

% Plot the average LSR
figure;
plot(avg_lsr);  % Plot the average LSR across runs for this combination
title('Average LSR across Runs');
xlabel('Time');
ylabel('LSR Mean');

%% Example: Plot all LSR means for CF = 125 Hz, tone = 2500 Hz, db = 70 across all runs
figure;
hold on;
for run_i = 1:n_runs
    % Extract the LSR mean for each run
    lsr_run = lsr_means{cf_idx, tone_idx, db_idx, run_i};
    plot(lsr_run);
end
title('LSR Means Across Runs');
xlabel('Time');
ylabel('LSR Mean');
legend(arrayfun(@(x) sprintf('Run %d', x), 1:n_runs, 'UniformOutput', false));

%% % Create a struct to hold results by name
lsr_results = struct();

for i_cf = 1:length(cfs)
    for i_tonefreq = 1:length(frequencies)
        for i_db = 1:length(dbs)
            for run_i = 0:n_runs-1
                % Create a key based on CF, tone, db, and run
                key = sprintf('CF_%s_Tone_%s_dB_%d_Run_%d', cfs{i_cf}, frequencies{i_tonefreq}, dbs(i_db), run_i + 1);

                % Store the result in the struct
                lsr_results.(key) = lsr_means{i_cf, i_tonefreq, i_db, run_i + 1};
            end
        end
    end
end

% Access the result for CF = 125 Hz, Tone = 2500 Hz, dB = 70, Run 1:
result_key = 'CF_125.0_Tone_2500.0_dB_70_Run_1';
specific_lsr = lsr_results.(result_key);  % 20000x1 vector


