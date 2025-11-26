
folder ="/home/ekim/PycharmProjects/subcorticalSTRF/results";

rates = [2, 4, 8, 16, 32, 64, 128, 256, 512];
n_fiber_counts = rates;
n_files = length(rates);
tables = cell(1, n_files);

% Load tables from each file
for i = 1:n_files
    rate = rates(i);
    filename = sprintf('cf_tone_pair_across_runs_psth_table_%d-%d-%d.mat', rate, rate, rate);
    filepath = fullfile(folder, filename);

    S = load(filepath);
    tables{i} = S.T;
    % Adjust CF and Tone indices to be 1-based (if needed)
    tables{i}.("CF index") = tables{i}.("CF index") + 1;
    tables{i}.("Tone Index") = tables{i}.("Tone Index") + 1;
end

% Parameters
fixed_CF_val = 2500;
tone_freqs = [125, 332.8766, 700.4736, 1350.5111, 2500];
db_levels = [50, 60, 70, 80];
tol = 1e-3;

% Plot settings
colors = {'b', [0.85 0.33 0.1], 'g'};  % LSR, MSR, HSR
markers = {'o', 's', 'd'};
anf_types = {'LSR', 'MSR', 'HSR'};

figure('Position', [100, 100, 1400, 900]);
t = tiledlayout(length(db_levels), length(tone_freqs), 'Padding', 'compact', 'TileSpacing', 'compact');

% Loop over dB levels and tone frequencies
for lvl = 1:length(db_levels)
    for fidx = 1:length(tone_freqs)
        nexttile;
        hold on;

        % Loop over ANF types
        for anf_i = 1:length(anf_types)
            mean_rates_vs_n = zeros(1, n_files);
            sem_rates_vs_n = zeros(1, n_files);

       
            % Get reference rate from the 512-fiber run 
            ref_table = tables{end}; % Last one is 513
            ref_rows = abs(ref_table.("CF value") - fixed_CF_val) < tol & ...
                        abs(ref_table.("Freq Value") - tone_freqs(fidx)) < 1 & ...
                        abs(ref_table.("Db Level") - db_levels(lvl)) < tol;

            ref_idx = find(ref_rows,1);
            if isempty(ref_idx)
                warning("Reference data missing at 512 fibers");
                continue;
            end

            raw_col = sprintf('PSTH Runs - %s', anf_types{anf_i});
            ref_psth_mat = ref_table.(raw_col){ref_idx};    % [n_runs x T]
            ref_rates = mean(ref_psth_mat, 2);      % Mean over time -> [n_runs x 1]
            ref_rate_mean = mean(ref_rates);        % Final normalization scalar

            % Now go through all sample counts
            for i = 1:n_files
                T = tables{i};

                % Select the row matching fixed CF, tone freq, and dB
                rows = abs(T.("CF value") - fixed_CF_val) < tol & ...
                        abs(T.("Freq Value") - tone_freqs(fidx)) < 1 & ...
                        abs(T.("Db Level") - db_levels(lvl)) < tol;

                idx = find(rows,1);
                if isempty(idx)
                    warning('Missing data for CF=%.1f, Tone=%.1f, Level=%d in file %d', ...
                        fixed_CF_val, tone_freqs(fidx), db_levels(lvl), rates(i));
                    mean_rates_vs_n(i) = NaN;
                    sem_rates_vs_n(i) = NaN;
                    continue;
                end

                psth_mat = T.(raw_col){idx};        % [n_runs x T]
                rates_per_run = mean(psth_mat, 2);  % average over time -> [n_runs x 1]
                norm_rates = rates_per_run / ref_rate_mean; % normalize to 512 fiber rate

                mean_rates_vs_n(i) = mean(norm_rates);
                sem_rates_vs_n(i) = std(norm_rates) / sqrt(length(norm_rates));

            end


            % Plot mean with SEM error bars
            errorbar(n_fiber_counts, mean_rates_vs_n, sem_rates_vs_n, ...
                'Color', colors{anf_i}, ...
                'Marker', markers{anf_i}, ...
                'DisplayName', anf_types{anf_i}, ...
                'LineWidth', 1.5);
        end

        xlabel('Number of ANF fibers sampled (log scale)');
        ylabel('Mean firing rate (spikes/s)');
        title(sprintf('CF = %.0f Hz, Tone = %.0f Hz, Level = %d dB', ...
              fixed_CF_val, tone_freqs(fidx), db_levels(lvl)));

        set(gca, 'XScale', 'log');
        xticks(n_fiber_counts);
        xticklabels(string(n_fiber_counts));
        xtickangle(60);
        grid on;
        set(gca, 'FontSize', 11);
        hold off;
    end
end

legend(anf_types, 'Location', 'northeastoutside');
