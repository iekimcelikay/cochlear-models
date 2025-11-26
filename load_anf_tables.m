rates = [2, 4, 8, 16, 32, 64, 128, 256, 512];
anf_data = cell(size(rates));

for i = 1:length(rates)
    rate = rates(i);
    filename = sprintf('cf_tone_pair_table_3x5x4_%d-%d-%d.mat', rate, rate, rate);
    S = load(filename);
    anf_data{i} = S.cf_tone_pair_data;
    anf_data{i}.("CF index") = anf_data{i}.("CF index") + 1;
    anf_data{i}.("Tone Index") = anf_data{i}.("Tone Index") + 1;
    T = anf_data{i}; % To make the coding easier and more readable. 
    cf_1{i} = T(T{:,1} ==1, :);
    cf_2{i} = T(T{:,1} ==2, :);
    cf_3{i} = T(T{:,1} == 3, :);
   
end



% tone index decibels: 50 -> 0,1,2,3,4 
% tone index decibels: 60 -> 5,6,7,8,9
% 70 -> 10,11,12,13,14
% 80 -> 15,16,17,18,19

group_size = 5; % 5 elements in each group
start_value = 1;
num_groups = 4;
% start_index = i * group_size ( + start_value);
% end_index = start_index + group_size - 1;

% This loop is to index the decibels. 

for i = 0:(num_groups - 1 )
    start_index = i * group_size + start_value; 
    end_index = start_index + group_size - 1; 
    index_range = start_index:end_index;
    disp(index_range);
end


% Your tables in a cell array:
tables = anf_data;
n_fiber_counts = rates;

% Parameters:
fixed_CF_val = 125;           % fixed CF
tone_freqs = [125, 332.8766, 700.4736, 1.350511108888547e+03, 2500];
db_levels = [50,60,70,80];

tol = 1e-3;
% Colors & markers for ANF types
colors = {'b', [0.85 0.33 0.1], 'g'};  % Blue, orange, green
markers = {'o', 's', 'd'};
anf_types = {'LSR', 'MSR', 'HSR'};

figure('Position', [100, 100, 1400, 900]);
t = tiledlayout(length(db_levels), length(tone_freqs), 'Padding', 'compact', 'TileSpacing', 'compact');

% Loop over tone frequencies and dB levels, plot in subplots
%plot_idx = 1;
for lvl = 1:length(db_levels)
    for fidx = 1:length(tone_freqs)
        nexttile;
        hold on;

        %subplot(length(db_levels), length(tone_freqs), plot_idx);
        for anf_i = 1:length(anf_types)
            mean_rates_vs_n = zeros(1, length(tables));
            for i = 1:length(tables)
                T = tables{i};
                rows = abs(T.("CF value") - fixed_CF_val) < tol & ...
                       abs(T.("Freq Value") - tone_freqs(fidx)) < 1 & ...
                       abs(T.("Db Level") -  db_levels(lvl)) < tol ;
                   
                col_name = sprintf('Mean PSTH - %d %s', n_fiber_counts(i), anf_types{anf_i});
                cell_vectors = T.(col_name)(rows);
                mean_rates_per_row = cellfun(@(x) mean(x), cell_vectors);
                mean_rates_vs_n(i) = mean(mean_rates_per_row);
            end
            
            plot(n_fiber_counts, mean_rates_vs_n, ...
                'Color', colors{anf_i}, ...
                'Marker', markers{anf_i}, ...
                'DisplayName', anf_types{anf_i}, ...
                'LineWidth', 1.5);
        end
        
        xlabel('Number of ANF fibers sampled ( log scale) ');
        ylabel('Mean firing rate (spikes/s)');
        title(sprintf('CF=%.02f Hz, Tone=%.02f Hz, Level=%d dB', fixed_CF_val, tone_freqs(fidx), db_levels(lvl)));
        set(gca, 'XScale', 'log');
        xticks(n_fiber_counts);
        xticklabels(arrayfun(@num2str, n_fiber_counts, 'UniformOutput', false));
        xtickangle(60);
        grid on;
        hold off;
        set(gca, 'FontSize', 11);
%        plot_idx = plot_idx + 1;
    end
end
legend(anf_types, 'Location', 'northeastoutside');





%%

% Prepare output
mean_rates_LSR = zeros(1, length(anf_data));
mean_rates_MSR = zeros(1, length(anf_data));
mean_rates_HSR = zeros(1, length(anf_data));

for i = 1:length(anf_data)
    T = anf_data{i};
    
    % Filter rows matching stimulus conditions 

    rows = (T.("CF index") ==  fixed_CF_idx) & ...
            (T.("Freq Value") == fixed_tone_freq) & ...
            (T.("Db Level") == fixed_level);

    % Column names for this number of fibers
    col_LSR = sprintf('Mean PSTH - %d LSR', n_fiber_counts(i));
    col_MSR = sprintf('Mean PSTH - %d MSR', n_fiber_counts(i));
    col_HSR = sprintf('Mean PSTH - %d HSR', n_fiber_counts(i));
    
    % LSR
    cell_vectors = T.(col_LSR)(rows);
    mean_rates_per_row = cellfun(@(x) mean(x), cell_vectors);
    mean_rates_LSR(i) = mean(mean_rates_per_row);

    % MSR
    cell_vectors = T.(col_MSR)(rows);
    mean_rates_per_row = cellfun(@(x) mean(x), cell_vectors);
    mean_rates_MSR(i) = mean(mean_rates_per_row);

    % HSR
    cell_vectors = T.(col_HSR)(rows);
    mean_rates_per_row = cellfun(@(x) mean(x), cell_vectors);
    mean_rates_HSR(i) = mean(mean_rates_per_row);
end

% Plotting
figure; hold on;
plot(n_fiber_counts, mean_rates_LSR, '-o', 'DisplayName', 'LSR');
plot(n_fiber_counts, mean_rates_MSR, '-s', 'DisplayName', 'MSR');
plot(n_fiber_counts, mean_rates_HSR, '-d', 'DisplayName', 'HSR');

% Set x-axis ticks to your exact sample counts
xticks(n_fiber_counts);

% Optional: format x-axis tick labels as strings
xticklabels(arrayfun(@num2str, n_fiber_counts, 'UniformOutput', false));

xlabel('Number of ANF fibers sampled');
ylabel('Mean firing rate (spikes/s)');
title(sprintf('Mean rate vs ANF sample size\nCF=%d Hz, Tone=%d Hz, Level=%d dB SPL', ...
    fixed_CF_val, fixed_tone_freq, fixed_level));
legend('Location', 'best');
grid on;
hold off;