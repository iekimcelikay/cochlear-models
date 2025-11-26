function avg_psth = average_psth_across_runs(psth_data, fiber_type)
    % Inputs:
    %   psth_data: struct with fields .lsr, .msr, .hsr
    %   fiber_type: string, one of 'lsr', 'msr', or 'hsr'
    %
    % Output:
    %   avg_psth: cell array (num_cf x num_tone x num_db), each cell is
    %             average PSTH vector across runs

    % Get field from psth_data
    fiber_psths = psth_data.(fiber_type);

    % Get dimensions
    dims = size(fiber_psths);
    num_cf = dims(1);
    num_tone = dims(2);
    num_db = dims(3);
    num_runs = dims(4);

    avg_psth = cell(num_cf, num_tone, num_db);

    for i_cf = 1:num_cf
        for i_tone = 1:num_tone
            for i_db = 1:num_db
                % Collect all run PSTHs for this condition
                run_psths = fiber_psths{i_cf, i_tone, i_db, :};

                % Convert cell of vectors (1 per run) into matrix (cols = runs)
                run_psth_mat = cell2mat(cellfun(@(x) x(:), run_psths, 'UniformOutput', false));

                % Average across runs (along columns)
                avg_psth{i_cf, i_tone, i_db} = mean(run_psth_mat, 2);
            end
        end
    end
end
