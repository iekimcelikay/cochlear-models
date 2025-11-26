%% Plot cochlea mean firing rates
% Loads pre-computed mean rates and creates visualizations
% Run after compute_cochlea_mean_rates.m

clear; clc;

% Load the saved mean rates
input_dir = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate/out/condition_psths";
data_file = fullfile(input_dir, 'cochlea_mean_rates.mat');

assert(isfile(data_file), 'Mean rates file not found. Run compute_cochlea_mean_rates.m first.');
load(data_file);

% Get dimensions
nCF = length(unique_cf);
nTone = length(unique_tone);
nDB = length(unique_db);

fprintf('Loaded mean rates for:\n');
fprintf('  %d CFs: %.1f to %.1f Hz\n', nCF, min(unique_cf), max(unique_cf));
fprintf('  %d Tone freqs: %.1f to %.1f Hz\n', nTone, min(unique_tone), max(unique_tone));
fprintf('  %d dB levels: %s\n', nDB, mat2str(unique_db));

%% Optional: Plot per CF (rate vs tone frequency, one figure with subplots)
make_overview = true;

if make_overview
    figure('Name','Mean Rates (All CFs)','Color','w','Position',[100 100 2400 1200]);  % Much wider figure
    
    % Calculate optimal layout dimensions
    num_cols = 5;
    num_rows = ceil(nCF/num_cols);
    
    % Create fixed-size subplots with more space for x-axis labels
    t = tiledlayout(num_rows, num_cols, 'TileSpacing', 'compact', 'Padding', 'none');
    
    % Color settings
    colormap_name = 'parula'; % Options: 'parula', 'jet', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', etc.
    cmap = eval([colormap_name '(numel(fiber_types))']);
    
    for icf = 1:nCF
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
            db_norm = (unique_db - min(unique_db)) / (max(unique_db) - min(unique_db));
            
            % Collapse over dB: show each dB as separate line
            for idb = 1:nDB
                y = squeeze(mean_rates.(ft)(icf, :, idb));
                if all(isnan(y)), continue; end
                
                % More subtle color adjustment - maintain color scheme better
                % Adjust darkness without significant hue shifts
                darkness_factor = 0.58 + 0.32 * db_norm(idb); % 0.6-1.0 range (lighter to darker)
                
                % Simple color darkening (preserve hue better)
                color = base_color * darkness_factor;
                
                % Increase line width differentiation 
                line_width = 1.0 + 2.5 * db_norm(idb); % 1.0-3.5 range (more variation)
                
                % Add line style variation for lowest dB levels if needed
                if idb == 1 && nDB > 3
                    linestyle = '--'; % Dashed line for lowest dB
                else
                    linestyle = '-';  % Solid line for others
                end
                
                plot(unique_tone, y, 'Color', color, 'LineStyle', linestyle, ...
                     'LineWidth', line_width, 'DisplayName', sprintf('%s %gdB', upper(ft), unique_db(idb)));
            end
        end
        title(sprintf('CF %.0f Hz', unique_cf(icf)), 'FontSize', 9);
        
        % Use linear scale 
        set(gca,'XScale','linear');
        
        % Show ALL frequency values on x-axis as requested
        xticks(unique_tone);
        xticklabels(arrayfun(@(x) sprintf('%.0f', x), unique_tone, 'UniformOutput', false));
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
            all_y_data = [all_y_data; mean_rates.(ft{1})(icf, :, :)];
        end
        ymax = max(all_y_data(:), [], 'omitnan') * 1.05; % Add 5% margin
        ylim([0, ymax]);
    end
    
    % Add common x and y labels for the entire figure
    xlabel(t, 'Tone Frequency (Hz)', 'FontSize', 12);
    ylabel(t, 'Mean Rate (spikes/s)', 'FontSize', 12);
    
    % Create a single legend for the entire figure
    leg = legend('show', 'FontSize', 8, 'NumColumns', 3);
    leg.Layout.Tile = 'south'; % Place legend at bottom of all subplots
    
    sgtitle('Mean Firing Rate vs Tone Frequency (Single Run Cochlea)', 'FontSize', 14);
    
    % Save the overview plot with higher resolution
    saveas(gcf, fullfile(input_dir, 'cochlea_tuning_overview.png'));
    print(fullfile(input_dir, 'cochlea_tuning_overview_highres.png'), '-dpng', '-r300');
    fprintf('Saved overview plot\n');
end

%% Optional: Single CF detailed plot (edit cf_target)
cf_target = 1000; % Set to a real CF value to generate this plot
if ~isempty(cf_target)
    [~, idxCF] = min(abs(unique_cf - cf_target));

    figure('Color','w', 'Position', [100 100 1000 600]); hold on;
    
    % Color settings - use same colormap as overview plot
    colormap_name = 'winter'; % Match to overview plot
    cmap = eval([colormap_name '(numel(fiber_types))']);
    
    % Create color and line width scaling based on dB level
    db_norm = (unique_db - min(unique_db)) / (max(unique_db) - min(unique_db));
    
    for fti = 1:numel(fiber_types)
        ft = fiber_types{fti};
        base_color = cmap(fti,:);
        
        for idb = 1:nDB
            y = squeeze(mean_rates.(ft)(idxCF, :, idb));
            if all(isnan(y)), continue; end
            
            % More subtle color adjustment - maintain color scheme better
            % Adjust darkness without significant hue shifts
            darkness_factor = 0.6 + 0.4 * db_norm(idb); % 0.6-1.0 range (lighter to darker)
            
            % Simple color darkening (preserve hue better)
            color = base_color * darkness_factor;
            
            % Increase line width differentiation
            line_width = 1.2 + 2.8 * db_norm(idb); % 1.2-4.0 range (more variation)
            
            % Add line style variation for lowest dB levels if needed
            if idb == 1 && nDB > 3
                linestyle = '--'; % Dashed line for lowest dB
            else
                linestyle = '-';  % Solid line for others
            end
            
            plot(unique_tone, y, 'Color', color, 'LineStyle', linestyle, ...
                 'LineWidth', line_width, 'DisplayName', sprintf('%s %gdB', upper(ft), unique_db(idb)));
        end
    end
    
    % Use linear scale instead of logarithmic
    set(gca,'XScale','linear');
    
    % Show ALL frequency values on x-axis
    xticks(unique_tone);
    xticklabels(arrayfun(@(x) sprintf('%.0f', x), unique_tone, 'UniformOutput', false));
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
        all_y_data = [all_y_data; mean_rates.(ft{1})(idxCF, :, :)];
    end
    ymax = max(all_y_data(:), [], 'omitnan') * 1.05; % Add 5% margin
    ylim([0, ymax]);
    
    xlabel('Tone Frequency (Hz)', 'FontSize', 10);
    ylabel('Mean Firing Rate (spikes/s)', 'FontSize', 10);
    title(sprintf('CF %.0f Hz', unique_cf(idxCF)), 'FontSize', 12);
    legend('Location','best', 'FontSize', 8);
    grid on;
    
    % Optional: Add a vertical line at the CF frequency for reference
    xline(unique_cf(idxCF), '--k', 'CF', 'LineWidth', 1, 'Alpha', 0.7);
    
    % Save the single CF plot
    saveas(gcf, fullfile(input_dir, sprintf('cochlea_tuning_CF%.0fHz.png', unique_cf(idxCF))));
    fprintf('Saved single CF plot for CF=%.0f Hz\n', unique_cf(idxCF));
end

%% Generate a comparative plot of tuning curves for selected CFs
cf_targets = [250, 500, 1000, 2000]; % Example CFs to plot
db_idx = 3; % Choose a specific dB level for comparison (adjust as needed)

if exist('db_idx', 'var') && db_idx <= nDB
    figure('Color','w', 'Position', [100 100 1200 800]); 
    tiledlayout(2, 2, 'TileSpacing', 'compact');

    colormap_name = 'winter';
    cmap = eval([colormap_name '(numel(fiber_types))']);

    for i = 1:length(cf_targets)
        cf_target = cf_targets(i);
        [~, idxCF] = min(abs(unique_cf - cf_target));
        
        nexttile; hold on;
        
        for fti = 1:numel(fiber_types)
            ft = fiber_types{fti};
            y = squeeze(mean_rates.(ft)(idxCF, :, db_idx));
            
            plot(unique_tone, y, 'Color', cmap(fti,:), 'LineWidth', 2, ...
                     'DisplayName', sprintf('%s', upper(ft)));
        end
        
        set(gca,'XScale','linear');
        xticks(unique_tone);
        xticklabels(arrayfun(@(x) sprintf('%.0f', x), unique_tone, 'UniformOutput', false));
        xtickangle(90);
        
        title(sprintf('CF = %.0f Hz', unique_cf(idxCF)));
        xlabel('Tone Frequency (Hz)');
        ylabel('Mean Firing Rate (spikes/s)');
        xline(unique_cf(idxCF), '--k', 'LineWidth', 1);
        grid on;
        
        % Only add legend to first subplot
        if i == 1
            legend('Location', 'best');
        end
    end

    sgtitle(sprintf('Cochlea Tuning Curves at %d dB', unique_db(db_idx)), 'FontSize', 14);
    saveas(gcf, fullfile(input_dir, sprintf('cochlea_tuning_comparison_%ddB.png', unique_db(db_idx))));
    fprintf('Saved comparative tuning plot\n');
end

fprintf('All cochlea tuning plots completed.\n');