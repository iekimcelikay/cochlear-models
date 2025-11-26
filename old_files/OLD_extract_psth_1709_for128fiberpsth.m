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
                data = load(filepath);
                results = data.synapse_result_struct;

                % --- DEBUG: check if PSTH is empty and fiber types present ---
                if isempty(results.psth)
                    fprintf('WARNING: Empty PSTH at CF %g, Tone %g, dB %d, Run %d\n', ...
                            cf_val_num, tone_val_num, db_val, run_i+1);
                        % store indices instead of printing
                    empty_cells(end+1,:) = [i_cf, i_tone, i_db, run_i+1];
                end
                
                unique_types = unique(results.anf_type);
                fprintf('Fiber types in file: %s\n', strjoin(unique_types(:)', ', '));
                % Also check if any fiber types are missing
                fiber_types = {'lsr','msr','hsr'};
                for ft = fiber_types
                    if ~any(strcmp(results.anf_type, ft{1}))
                        % store as negative run index to indicate missing fiber type (optional)
                        empty_cells(end+1,:) = [i_cf, i_tone, i_db, run_i+1];  
                    end
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

save('psth_data_128fibers.mat', 'lsr_all', 'msr_all', 'hsr_all', 'cfs', 'frequencies', 'dbs');



%% Summary chatgpt given nneeds to be fixed.
fprintf('\n=== SUMMARY ===\n');
for ft = fiber_types
    ft_name = ft{1};
    total_cells = num_cf * numel(frequencies) * numel(dbs) * n_runs;
    filled = fiber_counts.(ft_name);
    fprintf('%s: filled %d / %d cells (%.1f%%)\n', ...
        upper(ft_name), filled, total_cells, 100*filled/total_cells);
end

%% COMPUTE MEAN ± SEM ACROSS RUNS

all_data = struct(...
    'lsr', lsr_all,...
    'msr', msr_all,...
    'hsr', hsr_all);

mean_rates = struct('lsr', zeros(num_cf, length(frequencies), length(dbs)), ...
                    'msr', zeros(num_cf, length(frequencies), length(dbs)), ...
                    'hsr', zeros(num_cf, length(frequencies), length(dbs)));
sem_rates = mean_rates;
n_runs = 10;
num_cf = 20;

for i_cf = 1:num_cf
    for i_tone = 1:length(frequencies)
        for i_db = 1:length(dbs)
            for field = ["lsr", "msr", "hsr"]
                % Collect data across runs
                runs = zeros([], n_runs); % each run should return a vector 

                for i_run = 1:n_runs
                    data = all_data(i_cf, i_tone, i_db, i_run).(field);
                    % Preallocate the matrix on first run
                    if i_run == 1
                        % Preallocate the matrix on first run
                        runs = zeros(length(data), n_runs);
                    end
                    runs(:, i_run) = data(:); % Ensure column vector 
                end
                % Now runs is a [time_bins x n_runs] matrix
                mean_psth = mean(runs, 2); % Average across runs, per time bin
                mean_rates.(field)(i_cf, i_tone, i_db) = mean(mean_psth);

                mean_rate_per_run = mean(runs, 1); % Average per run
                sem_rates.(field)(i_cf, i_tone, i_db) = std(mean_rate_per_run)/sqrt(n_runs);
            end
        end
    end
end

save('mean_rates_128fibers.mat', 'mean_rates', 'sem_rates');



%% PLOT RESULTS
fiber_type_fields = {'lsr', 'msr', 'hsr'};
fiber_type_names = {'LSR', 'MSR', 'HSR'};
colors = lines(3);

for i_cf = 1:num_cf
    cf_val = str2double(cfs{i_cf});
    figure('Name', sprintf('CF = %g Hz', cf_val), 'NumberTitle','off');
    sgtitle(sprintf('Mean Firing Rate (128 Fibers) – CF = %g Hz', cf_val));

    t = tiledlayout(length(dbs), length(frequencies), 'Padding','compact','TileSpacing','compact');

    for i_db = 1:length(dbs)
        for i_tone = 1:length(frequencies)
            ax = nexttile;
            hold(ax, 'on');
            for ftype_i = 1:length(fiber_type_fields)
                field = fiber_type_fields{ftype_i};
                mean_rate = mean_rates.(field)(i_cf, i_tone, i_db);
                sem = sem_rates.(field)(i_cf, i_tone, i_db);
                bar(ax, ftype_i, mean_rate, 'FaceColor', colors(ftype_i,:));
                errorbar(ax, ftype_i, mean_rate, sem, 'k', 'LineWidth', 1.2);
            end
            xticks(ax, 1:3); xticklabels(ax, fiber_type_names);
            ylabel(ax, 'Mean firing rate (spk/s)');
            title(ax, sprintf('%s Hz – %d dB', frequencies{i_tone}, dbs(i_db)));
            hold(ax,'off');
        end
    end
end


%% Plot mean firing rate vs tone frequency(for a given CF and dB level)

cf_idx = 10;        % pick a CF (1 to num_cf)
db_idx = 2;         % pick a level (1 to 4)

cfs = {'125.0', '159.49726883', '198.39318028', '242.24859198', '291.69587485', '347.44803149', ...
    '410.30897734', '481.18513264', '561.09849255', '651.20136376', '752.79298009', '867.33823676', ...
    '996.48881333', '1142.10699008', '1306.29250097', '1491.41281075', ...
    '1700.13725242', '1935.4755176', '2200.82105458', '2500.0'};
cf_freq = str2double(cfs{cf_idx});

frequencies = cfs;

fields = ["lsr", "msr", "hsr"];
colors = lines(numel(fields));
frequencies = cellfun(@str2double, frequencies);

figure; clf; hold on;
for f = 1:numel(fields)
    field = fields(f);
    field_char = char(field); 

    y = squeeze(mean_rates.(field_char)(cf_idx, :, db_idx));
    err = squeeze(sem_rates.(field_char)(cf_idx, :, db_idx));
    
    errorbar(frequencies, y, err, 'Color', colors(f,:), ...
        'DisplayName', field_char, 'LineWidth', 1.5);
end

xlabel('Tone Frequency (Hz)');
ylabel('Mean Firing Rate (spikes/s)');
title(sprintf('CF #%d = %.0f Hz - Level = %d dB', cf_idx,cf_freq, dbs(db_idx)));
legend;
grid on;

%%
% Parameters
num_cf = numel(cfs);
cf_vals = cellfun(@str2double, cfs);       % numeric CFs
frequencies = cf_vals;                     % stimulus frequencies
num_dbs = length(dbs);
fields = ["lsr", "msr", "hsr"];
colors = lines(numel(fields));
line_styles = {'-', '--', ':', '-.'};      % for different dB levels

% Set up figure
figure('Name', 'Mean Rates: All CFs, All dBs', 'Color', 'w', 'Position', [100, 100, 1400, 800]);
tiledlayout(4, 5, 'TileSpacing', 'compact');  % 20 subplots (4x5 grid)

for cf_idx = 1:num_cf
    nexttile;
    cf_freq = cf_vals(cf_idx);

    for db_idx = 1:num_dbs
        for f = 1:numel(fields)
            field = fields(f);
            field_char = char(field); 

            y = squeeze(mean_rates.(field_char)(cf_idx, :, db_idx));
            err = squeeze(sem_rates.(field_char)(cf_idx, :, db_idx));

            % Slightly shift color or use linestyle to differentiate levels
            plot_style = line_styles{db_idx};
            color = colors(f,:);
            
            errorbar(frequencies, y, err, ...
                'LineStyle', plot_style, ...
                'Color', color, ...
                'LineWidth', 1.2, ...
                'DisplayName', sprintf('%s - %ddB', field_char, dbs(db_idx)));
            hold on;
        end
    end

    title(sprintf('CF #%d\n%.0f Hz', cf_idx, cf_freq), 'FontSize', 9);
    set(gca, 'XScale', 'log');
    xlim([min(frequencies), max(frequencies)]);
    ylim auto;
    grid on;

    if cf_idx > 15
        xlabel('Freq (Hz)');
    end
    if mod(cf_idx,5) == 1
        ylabel('Rate (spikes/s)');
    end
end

% Add main figure title
sgtitle('Mean Firing Rate vs Frequency for All CFs and Levels', 'FontSize', 14);

% Add legend outside plot area
lg = legend('Location', 'eastoutside');
lg.Layout.Tile = 'east';  % Attach to side of tiled layout
%%
% Setup
cfs = {'125.0', '159.49726883', '198.39318028', '242.24859198', '291.69587485', ...
    '347.44803149', '410.30897734', '481.18513264', '561.09849255', '651.20136376', ...
    '752.79298009', '867.33823676', '996.48881333', '1142.10699008', '1306.29250097', ...
    '1491.41281075', '1700.13725242', '1935.4755176', '2200.82105458', '2500.0'};
cf_vals = cellfun(@str2double, cfs);

fields = ["lsr", "msr", "hsr"];
colors = lines(numel(fields));

% Define line styles and markers per dB level
dbs = [40, 50, 60, 70];  % example dB levels; replace with your 'dbs' variable if different
line_styles = {'-', '--', ':', '-.'};
markers = {'o', 's', '^', 'd'};

num_cf = length(cf_vals);
frequencies = cf_vals;  % or your stimulus frequencies if different

figure('Name', 'Mean Firing Rates per CF with Multiple dB levels', ...
    'Color', 'w', 'Position', [100, 100, 1400, 900]);
tiledlayout(4, 5, 'TileSpacing', 'compact', 'Padding', 'compact');  % 20 CFs in 4x5 grid

for cf_idx = 1:num_cf
    nexttile;
    hold on;

    cf_freq = cf_vals(cf_idx);

    for db_idx = 1:length(dbs)
        for f = 1:numel(fields)
            field_char = char(fields(f));
            
            y = squeeze(mean_rates.(field_char)(cf_idx, :, db_idx));
            err = squeeze(sem_rates.(field_char)(cf_idx, :, db_idx));
            
            errorbar(frequencies, y, err, ...
                'LineStyle', line_styles{db_idx}, ...
                'Marker', markers{db_idx}, ...
                'Color', colors(f,:), ...
                'LineWidth', 1.2, ...
                'MarkerSize', 5, ...
                'DisplayName', sprintf('%s %ddB', field_char, dbs(db_idx)));
        end
    end

    title(sprintf('CF #%d = %.0f Hz', cf_idx, cf_freq), 'FontSize', 10);
    xlabel('Tone Frequency (Hz)');
    ylabel('Mean Firing Rate (spikes/s)');
    xlim([min(frequencies), max(frequencies)]);
    ylim auto;
    grid on;
    % Keep linear axis:
    % (No need to set XScale, linear is default)
end

sgtitle('Mean Firing Rate vs Frequency for Each CF and dB Level', 'FontSize', 16);

% Show one combined legend outside the tiled plots:
lg = legend('Location', 'eastoutside');
lg.Layout.Tile = 'east';  % attach legend on the right of the tiled layout


%%
cfs = {'125.0', '159.49726883', '198.39318028', '242.24859198', '291.69587485', ...
    '347.44803149', '410.30897734', '481.18513264', '561.09849255', '651.20136376', ...
    '752.79298009', '867.33823676', '996.48881333', '1142.10699008', '1306.29250097', ...
    '1491.41281075', '1700.13725242', '1935.4755176', '2200.82105458', '2500.0'};
cf_vals = cellfun(@str2double, cfs);


% Setup
dbs = [40, 50, 60, 70];
fields = ["lsr", "msr", "hsr"];

cf_idx = 10;
frequencies = cf_vals;  % Your frequency vector

% Pick a nicer colormap for fiber types (3 distinct colors)
% You can try 'parula', 'viridis', or specify manually
base_colors = cool(numel(fields)); % 3 colors, visually appealing

line_styles = {'-', '-', '-', '-'}; % keep line style uniform for clarity
markers = {'o', 'o', 'o', 'o'};     % uniform marker shape for now

% Normalize dB for saturation scale between 0.3 and 1 (avoid zero saturation)
db_min = min(dbs);
db_max = max(dbs);
sat_levels = 0.3 + 0.7 * (dbs - db_min) / (db_max - db_min);

% Normalize dB for line width between 1 and 3
lw_min = 1;
lw_max = 3;
line_widths = lw_min + (lw_max - lw_min) * (dbs - db_min) / (db_max - db_min);

figure; clf; hold on;

for f = 1:numel(fields)
    field_char = char(fields(f));
    base_rgb = base_colors(f,:);  % e.g. [0.1 0.6 0.8]
    
    % Convert base RGB to HSV
    base_hsv = rgb2hsv(base_rgb);
    
    for db_idx = 1:length(dbs)
        y = squeeze(mean_rates.(field_char)(cf_idx, :, db_idx));
        err = squeeze(sem_rates.(field_char)(cf_idx, :, db_idx));
        
        % Adjust saturation only, keep hue and value constant
        new_hsv = base_hsv;
        new_hsv(2) = sat_levels(db_idx);
        
        % Convert back to RGB
        plot_color = hsv2rgb(new_hsv);
        
        errorbar(frequencies, y, err, ...
            'Color', plot_color, ...
            'LineStyle', line_styles{db_idx}, ...
            'Marker', markers{db_idx}, ...
            'MarkerSize', 7, ...
            'LineWidth', line_widths(db_idx), ...
            'DisplayName', sprintf('%s %ddB', field_char, dbs(db_idx)));
    end
end

xlabel('Tone Frequency (Hz)');
ylabel('Mean Firing Rate (spikes/s)');
title(sprintf('CF #%d = %.0f Hz', cf_idx, cf_vals(cf_idx)));
legend('Location', 'bestoutside');
grid on;
xlim([min(frequencies) max(frequencies)]);

%%
cfs = {'125.0', '159.49726883', '198.39318028', '242.24859198', '291.69587485', ...
    '347.44803149', '410.30897734', '481.18513264', '561.09849255', '651.20136376', ...
    '752.79298009', '867.33823676', '996.48881333', '1142.10699008', '1306.29250097', ...
    '1491.41281075', '1700.13725242', '1935.4755176', '2200.82105458', '2500.0'};
cf_vals = cellfun(@str2double, cfs);

% Setup
dbs = [40, 50, 60, 70];
fields = ["lsr", "msr", "hsr"];

frequencies = cf_vals;  % Your frequency vector

% Pick a nicer colormap for fiber types (3 distinct colors)
base_colors = cool(numel(fields)); % visually appealing colors

line_styles = {'-', '-', '-', '-'}; % uniform line style
markers = {'o', 'o', 'o', 'o'};     % uniform marker shape

% Normalize dB for saturation scale between 0.3 and 1 (avoid zero saturation)
db_min = min(dbs);
db_max = max(dbs);
sat_levels = 0.3 + 0.7 * (dbs - db_min) / (db_max - db_min);

% Normalize dB for line width between 1 and 3
lw_min = 1;
lw_max = 3;
line_widths = lw_min + (lw_max - lw_min) * (dbs - db_min) / (db_max - db_min);

for cf_idx = 1:numel(cf_vals)
    fig = figure('Visible','off', 'Position', [100 100 1200 800]); clf; hold on;  % invisible figure for speed
    
    for f = 1:numel(fields)
        field_char = char(fields(f));
        base_rgb = base_colors(f,:);
        base_hsv = rgb2hsv(base_rgb);
        
        for db_idx = 1:length(dbs)
            y = squeeze(mean_rates.(field_char)(cf_idx, :, db_idx));
            err = squeeze(sem_rates.(field_char)(cf_idx, :, db_idx));
            
            new_hsv = base_hsv;
            new_hsv(2) = sat_levels(db_idx);  % adjust saturation
            
            plot_color = hsv2rgb(new_hsv);
            
            errorbar(frequencies, y, err, ...
                'Color', plot_color, ...
                'LineStyle', line_styles{db_idx}, ...
                'Marker', markers{db_idx}, ...
                'MarkerSize', 7, ...
                'LineWidth', line_widths(db_idx), ...
                'DisplayName', sprintf('%s %ddB', field_char, dbs(db_idx)));
        end
    end
    
    xlabel('Tone Frequency (Hz)');
    ylabel('Mean Firing Rate (spikes/s)');
    title(sprintf('CF #%d = %.0f Hz', cf_idx, cf_vals(cf_idx)));
    legend('Location', 'northeastoutside');
    grid on;
    xlim([min(frequencies) max(frequencies)]);
    ax = gca;
    ax.Position = [0.1 0.1 0.75 0.8];

    
    % Save has PDF
    filename = sprintf('CF_%02d_%.0fHz.pdf', cf_idx, cf_vals(cf_idx));
    print(gcf, filename, '-dpdf', '-bestfit');
    exportgraphics(fig, filename, 'ContentType', 'vector', 'Resolution', 300);

    
    close(fig); % Close figure after saving
end

