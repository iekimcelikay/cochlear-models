%% Compute mean rates and SEM from extracted PSTHs
% This script loads pre-extracted PSTHs, computes mean rates and SEM, and creates plots
% Run this after bez_extract_psth.m

clear; clc;


%% Load extracted PSTH data
folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate";
output_dir = fullfile(folder, 'results/processed_data');
psth_data_file = fullfile(output_dir, 'psth_data_128fibers.mat');

fprintf('Loading PSTH data from %s\n', psth_data_file);
assert(isfile(psth_data_file), 'PSTH data file not found. Run bez_extract_psth.m first.');

% Load the saved data, which should include n_runs
load(psth_data_file);

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
fprintf('  %d Tone freqs: %.1f to %.1f Hz\n', length(frequencies), min(cf_vals), max(cf_vals));
fprintf('  %d dB levels: %s\n', length(dbs), mat2str(dbs));
fprintf('  %d runs per condition\n', n_runs);


%% COMPUTE MEAN ± SEM ACROSS RUNS
mean_rates = struct('lsr', zeros(num_cf, length(frequencies), length(dbs)), ...
                    'msr', zeros(num_cf, length(frequencies), length(dbs)), ...
                    'hsr', zeros(num_cf, length(frequencies), length(dbs)));
sem_rates = mean_rates;

%all_data = struct('lsr', lsr_all, 'msr', msr_all, 'hsr', hsr_all);

for i_cf = 1:num_cf
    for i_tone = 1:length(frequencies)
        for i_db = 1:length(dbs)
            for ft_idx = 1:length(fiber_types)
                field = fiber_types{ft_idx};
                
                 % Access the appropriate cell array directly based on fiber type
                if strcmp(field, 'lsr')
                    run_data = lsr_all(i_cf, i_tone, i_db, :);
                elseif strcmp(field, 'msr')
                    run_data = msr_all(i_cf, i_tone, i_db, :);
                elseif strcmp(field, 'hsr')
                    run_data = hsr_all(i_cf, i_tone, i_db, :);
                end

                 % Reshape to column vector
                run_data = reshape(run_data, [], 1);
                
                % Skip if no data
                if all(cellfun(@isempty, run_data))
                    mean_rates.(field)(i_cf, i_tone, i_db) = NaN;
                    sem_rates.(field)(i_cf, i_tone, i_db) = NaN;
                    continue;
                end
                
                % Get mean rate for each run
                run_means = zeros(1, n_runs);
                valid_runs = 0;
                
                for i_run = 1:n_runs
                    if ~isempty(run_data{i_run})
                        run_means(i_run) = mean(run_data{i_run});
                        valid_runs = valid_runs + 1;
                    else
                        run_means(i_run) = NaN;
                    end
                end
                
                % Average across valid runs
                if valid_runs > 0
                    mean_rates.(field)(i_cf, i_tone, i_db) = nanmean(run_means);
                    sem_rates.(field)(i_cf, i_tone, i_db) = nanstd(run_means) / sqrt(valid_runs);
                else
                    mean_rates.(field)(i_cf, i_tone, i_db) = NaN;
                    sem_rates.(field)(i_cf, i_tone, i_db) = NaN;
                end
            end
        end
    end
    
    fprintf('Computed mean rates for CF %d of %d\n', i_cf, num_cf);
end

% Save computed mean rates
save(fullfile(output_dir, 'mean_rates_128fibers.mat'), 'mean_rates', 'sem_rates', 'cfs', 'frequencies', 'dbs');
fprintf('Saved mean rates to %s\n', fullfile(output_dir, 'mean_rates_128fibers.mat'));

%% Create overview plot with consistent formatting
figure('Name','Mean Rates (All CFs) - BEZ2018','Color','w','Position',[100 100 2400 1200]);

% Calculate optimal layout dimensions
num_cols = 5;
num_rows = ceil(num_cf/num_cols);

% Create fixed-size subplots with more space for x-axis labels
t = tiledlayout(num_rows, num_cols, 'TileSpacing', 'compact', 'Padding', 'none');

% Color settings
colormap_name = 'winter'; % Options: 'parula', 'jet', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', etc.
cmap = eval([colormap_name '(numel(fiber_types))']);

for icf = 1:num_cf
    ax = nexttile; hold on;
    
    % Adjust subplot margins to make room for labels
    p = get(ax, 'Position');
    p(4) = p(4) * 0.85; % Reduce height to 85%
    set(ax, 'Position', p);
    
    for fti = 1:numel(fiber_types)
        ft = fiber_types{fti};
        base_color = cmap(fti,:);
        
        % Create color and line width scaling based on dB level
        % Normalize dB levels to 0-1 range for color scaling
        db_norm = (dbs - min(dbs)) / (max(dbs) - min(dbs));
        
        % Collapse over dB: show each dB as separate line
        for idb = 1:length(dbs)
            y = squeeze(mean_rates.(ft)(icf, :, idb));
            err = squeeze(sem_rates.(ft)(icf, :, idb));
            
            if all(isnan(y)), continue; end
            
            % More subtle color adjustment - maintain color scheme better
            % Adjust darkness without significant hue shifts
            darkness_factor = 0.6 + 0.4 * db_norm(idb); % 0.6-1.0 range (lighter to darker)
            
            % Simple color darkening (preserve hue better)
            color = base_color * darkness_factor;
            
            % Increase line width differentiation 
            line_width = 1.0 + 2.5 * db_norm(idb); % 1.0-3.5 range (more variation)
            
            % Add line style variation for lowest dB levels if needed
            if idb == 1 && length(dbs) > 3
                linestyle = '--'; % Dashed line for lowest dB
            else
                linestyle = '-';  % Solid line for others
            end
            
            % Plot with error bars
            errorbar(cf_vals, y, err, 'Color', color, 'LineStyle', linestyle, ...
                 'LineWidth', line_width, 'DisplayName', sprintf('%s %gdB', upper(ft), dbs(idb)));
        end
    end
    title(sprintf('CF %.0f Hz', cf_vals(icf)), 'FontSize', 9);
    
    % Use linear scale 
    set(gca,'XScale','linear');
    
    % Show ALL frequency values on x-axis as requested
    xticks(cf_vals);
    xticklabels(arrayfun(@(x) sprintf('%.0f', x), cf_vals, 'UniformOutput', false));
    xtickangle(90);
    
    % Optimize tick label appearance
    ax.FontSize = 6; % Slightly larger font for better readability
    ax.TickLength = [0.01 0.01]; % Shorter tick marks
    
    % Give more space for x-axis labels
    ax.TickDir = 'out'; % Move ticks outside to free up space
    
    % Only show y-axis on leftmost plots
    if mod(icf-1, num_cols) ~= 0
        ax.YTickLabel = {};
    end
    
    % Remove x and y labels from individual plots to save space
    xlabel('');
    ylabel('');
    
    grid on;
    
    % Set y-axis limits to be consistent across all subplots
    all_y_data = [];
    for ft = fiber_types
        ftn = ft{1};
        data = squeeze(mean_rates.(ftn)(icf, :, :));
        all_y_data = [all_y_data; data(:)]; %#ok<AGROW>
    end
    ymax = nanmax(all_y_data) * 1.05; % Add 5% margin
    ylim([0, ymax]);
end

% Add common x and y labels for the entire figure
xlabel(t, 'Tone Frequency (Hz)', 'FontSize', 12);
ylabel(t, 'Mean Rate (spikes/s)', 'FontSize', 12);

% Create a single legend for the entire figure
leg = legend('show', 'FontSize', 8, 'NumColumns', 3);
leg.Layout.Tile = 'south'; % Place legend at bottom of all subplots

sgtitle('Mean Firing Rate vs Tone Frequency (BEZ2018 Model)', 'FontSize', 14);

% Save the overview plot with higher resolution
saveas(gcf, fullfile(output_dir, 'bez_tuning_overview.png'));
print(fullfile(output_dir, 'bez_tuning_overview_highres.png'), '-dpng', '-r300');
fprintf('Saved overview plot\n');

%% Single CF detailed plots
% Generate detailed plots for a few selected CFs
cf_targets = [500, 1000, 2000]; % Example CFs to plot in detail

for i = 1:length(cf_targets)
    cf_target = cf_targets(i);
    [~, idxCF] = min(abs(cf_vals - cf_target));
    
    figure('Color','w', 'Position', [100 100 1000 600]); hold on;
    
    % Color settings - use same colormap as overview plot
    colormap_name = 'winter'; % Match to overview plot
    cmap = eval([colormap_name '(numel(fiber_types))']);
    
    % Create color and line width scaling based on dB level
    db_norm = (dbs - min(dbs)) / (max(dbs) - min(dbs));
    
    for fti = 1:numel(fiber_types)
        ft = fiber_types{fti};
        base_color = cmap(fti,:);
        
        for idb = 1:length(dbs)
            y = squeeze(mean_rates.(ft)(idxCF, :, idb));
            err = squeeze(sem_rates.(ft)(idxCF, :, idb));
            
            if all(isnan(y)), continue; end
            
            % More subtle color adjustment - maintain color scheme better
            % Adjust darkness without significant hue shifts
            darkness_factor = 0.6 + 0.4 * db_norm(idb); % 0.6-1.0 range (lighter to darker)
            
            % Simple color darkening (preserve hue better)
            color = base_color * darkness_factor;
            
            % Increase line width differentiation
            line_width = 1.2 + 2.8 * db_norm(idb); % 1.2-4.0 range (more variation)
            
            % Add line style variation for lowest dB levels if needed
            if idb == 1 && length(dbs) > 3
                linestyle = '--'; % Dashed line for lowest dB
            else
                linestyle = '-';  % Solid line for others
            end
            
            % Plot with error bars
            errorbar(cf_vals, y, err, 'Color', color, 'LineStyle', linestyle, ...
                 'LineWidth', line_width, 'DisplayName', sprintf('%s %gdB', upper(ft), dbs(idb)));
        end
    end
    
    % Use linear scale instead of logarithmic
    set(gca,'XScale','linear');
    
    % Show ALL frequency values on x-axis
    xticks(cf_vals);
    xticklabels(arrayfun(@(x) sprintf('%.0f', x), cf_vals, 'UniformOutput', false));
    xtickangle(90);
    
    % Ensure x-axis has enough padding for all labels
    curr_xlim = xlim;
    xlim([curr_xlim(1) - (curr_xlim(2)-curr_xlim(1))*0.05, curr_xlim(2) + (curr_xlim(2)-curr_xlim(1))*0.05]);
    
    % Adjust tick label appearance
    ax = gca;
    ax.FontSize = 8; % Font size for single plot
    ax.TickLength = [0.005 0.005]; % Shorter tick marks
    
    % Set y-axis limits based on data
    all_y_data = [];
    for ft = fiber_types
        ftn = ft{1};
        data = squeeze(mean_rates.(ftn)(idxCF, :, :));
        all_y_data = [all_y_data; data(:)]; %#ok<AGROW>
    end
    ymax = nanmax(all_y_data) * 1.05; % Add 5% margin
    ylim([0, ymax]);
    
    xlabel('Tone Frequency (Hz)', 'FontSize', 10);
    ylabel('Mean Firing Rate (spikes/s)', 'FontSize', 10);
    title(sprintf('CF %.0f Hz (BEZ2018 Model)', cf_vals(idxCF)), 'FontSize', 12);
    legend('Location','best', 'FontSize', 8);
    grid on;
    
    % Optional: Add a vertical line at the CF frequency for reference
    xline(cf_vals(idxCF), '--k', 'CF', 'LineWidth', 1, 'Alpha', 0.7);
    
    % Save the single CF plot
    saveas(gcf, fullfile(output_dir, sprintf('bez_tuning_CF%.0fHz.png', cf_vals(idxCF))));
    fprintf('Saved single CF plot for CF=%.0f Hz\n', cf_vals(idxCF));
end

fprintf('All BEZ2018 tuning plots completed.\n');