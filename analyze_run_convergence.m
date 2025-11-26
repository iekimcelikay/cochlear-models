% --- Configuration ---
folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF";
run_counts = [2, 4, 8, 16];  % Number of stimulus repetitions
n_files = length(run_counts);
fiber_str = '32-32-32';
tables = cell(1, n_files);

% Load tables from each file
for i = 1:n_files
    nruns = run_counts(i);
    filename = sprintf('cf_tone_pair_across_simulations_psth_table_%druns_%s.mat', nruns, fiber_str);
    filepath = fullfile(folder, filename);

    S = load(filepath);
    tables{i} = S.T;
    tables{i}.("CF index") = tables{i}.("CF index") + 1;
    tables{i}.("Tone Index") = tables{i}.("Tone Index") + 1;
end

% --- Parameters ---
fixed_CF_val = 125;
tone_freqs = [125, 332.8766, 700.4736, 1350.5111, 2500];
db_levels = [50, 60, 70, 80];
anf_types = {'LSR', 'MSR', 'HSR'};
tol = 1e-3;
convergence_eps = 0.05;
colors = {'b', [0.85 0.33 0.1], 'g'};
markers = {'o', 's', 'd'};
conv_n_mat = nan(length(db_levels), length(tone_freqs), length(anf_types));

% --- Plot setup ---
figure('Position', [100, 100, 1400, 900]);
t = tiledlayout(length(db_levels), length(tone_freqs), 'Padding', 'compact', 'TileSpacing', 'compact');
sgtitle(sprintf('Rate convergence across repetitions (Fixed CF = %.1f Hz, Fibers = %s)', ...
         fixed_CF_val, fiber_str), 'FontWeight', 'bold', 'FontSize', 14);

for lvl = 1:length(db_levels)
    for fidx = 1:length(tone_freqs)
        conv_text = cell(1, length(anf_types));
        nexttile;
        hold on;

        for anf_i = 1:length(anf_types)
            mean_rates_vs_n = zeros(1, n_files);
            sem_rates_vs_n = zeros(1, n_files);

            % Reference table is last one (16 runs)
            ref_table = tables{end};
            ref_rows = abs(ref_table.("CF value") - fixed_CF_val) < tol & ...
                       abs(ref_table.("Freq Value") - tone_freqs(fidx)) < 1 & ...
                       abs(ref_table.("Db Level") - db_levels(lvl)) < tol;
            ref_idx = find(ref_rows, 1);

            if isempty(ref_idx)
                warning("Reference data missing at 16 runs");
                continue;
            end

            raw_col = sprintf('PSTH Runs - %s', anf_types{anf_i});
            ref_psth_mat = ref_table.(raw_col){ref_idx};
            ref_rates = mean(ref_psth_mat, 2);
            ref_rate_mean = mean(ref_rates);

            for i = 1:n_files
                T = tables{i};
                rows = abs(T.("CF value") - fixed_CF_val) < tol & ...
                       abs(T.("Freq Value") - tone_freqs(fidx)) < 1 & ...
                       abs(T.("Db Level") - db_levels(lvl)) < tol;
                idx = find(rows, 1);

                if isempty(idx)
                    warning('Missing data for CF=%.1f, Tone=%.1f, Level=%d in run %d', ...
                        fixed_CF_val, tone_freqs(fidx), db_levels(lvl), run_counts(i));
                    mean_rates_vs_n(i) = NaN;
                    sem_rates_vs_n(i) = NaN;
                    continue;
                end

                psth_mat = T.(raw_col){idx};
                rates_per_run = mean(psth_mat, 2);
                norm_rates = rates_per_run / ref_rate_mean;

                mean_rates_vs_n(i) = mean(norm_rates);
                sem_rates_vs_n(i) = std(norm_rates) / sqrt(length(norm_rates));
            end

            % Convergence check
            abs_error = abs(mean_rates_vs_n - 1);
            conv_idx = find(abs_error < convergence_eps, 1);
            if ~isempty(conv_idx)
                conv_n = run_counts(conv_idx);
                conv_n_mat(lvl, fidx, anf_i) = conv_n;
                conv_text{anf_i} = sprintf('%s conv: %d runs', anf_types{anf_i}, conv_n);
            else
                conv_n_mat(lvl, fidx, anf_i) = NaN;
                conv_text{anf_i} = sprintf('%s: No conv.', anf_types{anf_i});
            end

            % Plot
            x = run_counts;
            y = mean_rates_vs_n;
            e = sem_rates_vs_n;

            fill([x fliplr(x)], [y - e fliplr(y + e)], ...
                colors{anf_i}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

            line_handles(anf_i) = plot(x, y, ...
                'Color', colors{anf_i}, ...
                'Marker', markers{anf_i}, ...
                'LineWidth', 1.5, ...
                'DisplayName', anf_types{anf_i});
        end

        xlabel('Number of stimulus repetitions (runs)');
        ylabel('Normalized mean rate');
        title(sprintf('CF = %.0f Hz, Tone = %.0f Hz, Level = %d dB', ...
              fixed_CF_val, tone_freqs(fidx), db_levels(lvl)));
        set(gca, 'XScale', 'log');
        xticks(run_counts);
        xticklabels(string(run_counts));
        xtickangle(60);
        grid on;
        set(gca, 'FontSize', 11);
        hold off;
    end
end

legend(line_handles, anf_types, 'Location', 'northeastoutside');

% --- Convergence Summary Window ---
fig = figure('Name', 'Run Convergence Summary', 'Position', [300, 300, 800, 500]);
txt = uicontrol('Style', 'edit', ...
    'Max', 2, 'Min', 0, ...
    'Units', 'normalized', ...
    'Position', [0 0 1 1], ...
    'HorizontalAlignment', 'left', ...
    'FontName', 'Courier New', ...
    'FontSize', 11);

summary_lines = {};
summary_lines{end+1} = sprintf('Convergence Summary for CF = %.0f Hz\n', fixed_CF_val);
summary_lines{end+1} = '-------------------------------------------';
summary_lines{end+1} = sprintf('%6s | %10s | %8s | %8s | %8s', ...
    'Level', 'ToneFreq', 'LSR', 'MSR', 'HSR');
summary_lines{end+1} = repmat('-', 1, 50);

out_rows = [];

for lvl = 1:length(db_levels)
    for fidx = 1:length(tone_freqs)
        row_str = sprintf('%6d | %10.1f |', db_levels(lvl), tone_freqs(fidx));
        row.Level_dB = db_levels(lvl);
        row.Tone_Hz = tone_freqs(fidx);
        row.Fibers = fiber_str;


        for anf_i = 1:length(anf_types)
            val = conv_n_mat(lvl, fidx, anf_i);
            if isnan(val)
                val_str = "No conv";
            else
                val_str = sprintf('%d', val);
            end
            row_str = [row_str, sprintf(' %8s |', val_str)];
            row.(anf_types{anf_i}) = val_str;
        end
        summary_lines{end+1} = row_str;
        out_rows = [out_rows; row];
    end
end

set(txt, 'String', summary_lines);

% --- Save .txt and .csv Summary ---
out_table = struct2table(out_rows);
base_filename = sprintf('convergence_summary_runs_CF_%dHz', fixed_CF_val);
outfile_txt = fullfile(folder, base_filename + ".txt");
outfile_csv = fullfile(folder, base_filename + ".csv");

writetable(out_table, outfile_txt, 'Delimiter', '\t');
writetable(out_table, outfile_csv);
fprintf('Convergence summary saved to:\n  %s\n  %s\n', outfile_txt, outfile_csv);
