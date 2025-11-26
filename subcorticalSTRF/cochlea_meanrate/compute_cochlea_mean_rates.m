%% Compute mean firing rates (single-run cochlea model)
% Uses PSTH files produced by cochlea_psth_extract_latest0051am.m
% Assumes each PSTH already averaged across fibers of same type for a CF/tone/db.
% No SEM (single run).

clear; clc;

input_dir = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate/out/condition_psths";
assert(isfolder(input_dir), 'Directory not found: %s', input_dir);

file_list = dir(fullfile(input_dir, 'psth_cf*_tone*_db*.mat'));
if isempty(file_list)
    error('No PSTH files found in %s', input_dir);
end

fiber_types = {'lsr','msr','hsr'};


% ---- Parse metadata from filenames ----
cf_vals = [];
tone_vals = [];
db_vals = [];

meta_table = struct('cf',{},'tone',{},'db',{},'file',{});

expr = '^psth_cf(?<cf>[-+]?\d*\.?\d+([eE][-+]?\d+)?)_tone(?<tone>[-+]?\d*\.?\d+([eE][-+]?\d+)?)_db(?<db>[-+]?\d*\.?\d+([eE][-+]?\d+)?)\.mat$';

tone_strings = {};

for k = 1:numel(file_list)
    fn = file_list(k).name;
    m = regexp(fn, expr, 'names');
    if isempty(m)
        warning('Skipping unrecognized file name: %s', fn);
        continue;
    end
    cf  = str2double(m.cf);
    tone= str2double(m.tone);
    db  = str2double(m.db);

    % Store original string representation for later comparison
    tone_strings{end+1} = m.tone; %#ok<AGROW>
    
    cf_vals(end+1)   = cf;   %#ok<AGROW>
    tone_vals(end+1) = tone; %#ok<AGROW>
    db_vals(end+1)   = db;   %#ok<AGROW>
    meta_table(end+1) = struct('cf',cf,'tone',tone,'db',db,'file',fullfile(file_list(k).folder, fn)); %#ok<AGROW>
end

if isempty(meta_table)
    error('No valid PSTH metadata parsed.');
end

% Unique sorted axes
unique_cf   = unique(cf_vals);
unique_tone = unique(tone_vals);
unique_db   = unique(db_vals);

% Debug: print all unique tone values to see exact stored values
fprintf('Unique tone values in files:\n');
disp(unique_tone);

% Debug: check for potential rounding issues
for i = 1:length(unique_tone)
    matching_strs = tone_strings(abs(tone_vals - unique_tone(i)) < 1e-10);
    if ~isempty(matching_strs)
        fprintf('Tone value %.10f appears in filenames as: %s\n', unique_tone(i), strjoin(unique(matching_strs), ', '));
    end
end


nCF   = numel(unique_cf);
nTone = numel(unique_tone);
nDB   = numel(unique_db);

% Preallocate mean_rates (spikes/s). NaN for missing.
mean_rates = struct();
for ft = fiber_types
    mean_rates.(ft{1}) = nan(nCF, nTone, nDB);
end

% Track fill counts
filled_counts = struct('lsr',0,'msr',0,'hsr',0);

% ---- Fill mean_rates ----
fprintf('Processing %d PSTH files...\n', numel(meta_table));
for k = 1:numel(meta_table)
    rec = meta_table(k);
    [~, i_cf]   = min(abs(unique_cf - rec.cf));
    [~, i_tone] = min(abs(unique_tone - rec.tone));
    [~, i_db]   = min(abs(unique_db - rec.db));

        % Debug: verify the match was correct
    if abs(unique_tone(i_tone) - rec.tone) > 1e-10
        fprintf('WARNING: Tone value mismatch. File has %.10f, matched to %.10f\n', rec.tone, unique_tone(i_tone));
    end

    S = load(rec.file, 'time', 'psth_struct');
    if ~isfield(S,'psth_struct')
        warning('Missing psth_struct in %s', rec.file);
        continue;
    end

    for ft = fiber_types
        ftn = ft{1};
        if isfield(S.psth_struct, ftn) && ~isempty(S.psth_struct.(ftn))
            psth_vec = S.psth_struct.(ftn)(:);  % column
            % Assume psth_vec already in spikes/s (bin rate). If counts per bin, uncomment:
            % binWidth = mean(diff(S.time));
            % mean_rate = sum(psth_vec) / (length(psth_vec)*binWidth);
            mean_rate = mean(psth_vec);  % average over time axis
            mean_rates.(ftn)(i_cf, i_tone, i_db) = mean_rate;
            filled_counts.(ftn) = filled_counts.(ftn) + 1;
        end
    end
end

% Add after processing files
for ft = fiber_types
    ftn = ft{1};
    missing_idx = find(isnan(mean_rates.(ftn)));
    if ~isempty(missing_idx)
        [i_cf, i_tone, i_db] = ind2sub(size(mean_rates.(ftn)), missing_idx);
        fprintf('Missing %s data points:\n', upper(ftn));
        for i = 1:min(10, length(i_cf))
            fprintf('  CF=%.1f, tone=%.1f, dB=%.1f\n', ...
                    unique_cf(i_cf(i)), unique_tone(i_tone(i)), unique_db(i_db(i)));
        end
        if length(i_cf) > 10
            fprintf('  ... and %d more\n', length(i_cf)-10);
        end
    end
end

% ---- Summary ----
total_cells = nCF * nTone * nDB;
fprintf('\n=== SUMMARY (single run, mean only) ===\n');
for ft = fiber_types
    ftn = ft{1};
    filled = filled_counts.(ftn);
    fprintf('%s: %d / %d (%.1f%%)\n', upper(ftn), filled, total_cells, 100*filled/total_cells);
end

% ---- Save ----
out_file = fullfile(input_dir, 'cochlea_mean_rates.mat');
save(out_file, 'mean_rates', 'unique_cf', 'unique_tone', 'unique_db', 'fiber_types', '-v7.3');
fprintf('Saved mean rates: %s\n', out_file);

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
    colormap_name = 'winter'; % Options: 'parula', 'jet', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', etc.
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
                darkness_factor = 0.6 + 0.4 * db_norm(idb); % 0.6-1.0 range (lighter to darker)
                
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