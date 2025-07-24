function synapse_results_fiber = run_fiber_synapse_batch_parfor(vihc,cf, cf_idx,freq, db, tone_idx, nrep, dt, noiseType, implnt, expliketype, fiber_filename)
    % Load fiber population

    fibers = load_AN_population_struct_by_type_then_cf(fiber_filename);
    cf_idx_target = cf_idx + 1;
    selected_fibers = fibers([fibers.cf_idx] == cf_idx_target);

    nfibers = numel(selected_fibers);

    fs_raw = 1 / dt;
    fs_target = 1000; % Hz
    downsample_factor = round(fs_raw / fs_target);

    % Preallocate afirrays (not struct fields!)
    fiber_idx_all  = zeros(1, nfibers);
    anf_type_all   = cell(1, nfibers);
    psth_all       = cell(1, nfibers);
    disp(class(cf));
    disp(cf);


    % === Parallel loop over fibers ===
    parfor idx_fiber = 1:nfibers
        spont = selected_fibers(idx_fiber).spont;
        tabs  = selected_fibers(idx_fiber).tabs;
        trel  = selected_fibers(idx_fiber).trel;
        fiber_cf = selected_fibers(idx_fiber).cf_idx;
        if iscell(fiber_cf)
            fiber_cf = fiber_cf{1};
        end

        fiber_idx = selected_fibers(idx_fiber).fiber_idx;
        if iscell(fiber_idx)
            fiber_idx = fiber_idx{1};
        end

        % Run synapse model
        psth_raw = model_Synapse_BEZ2018a(vihc, cf, nrep, dt, ...
            noiseType, implnt, spont, tabs, trel, expliketype);

        % Resample PSTH
        %if fs_target < fs_raw
         %   psth = decimate(double(psth_raw), downsample_factor);
        %else

        psth = single(psth_raw);
        %end

        % Store results in parallel-safe arrays

        fiber_idx_all(idx_fiber)  = fiber_idx;
        anf_type_all{idx_fiber}   = fibers(idx_fiber).anf_type;
        psth_all{idx_fiber}       = psth;

    end

    % === Repackage into struct (outside parfor) ===
    synapse_results_fiber.cf_idx     = cf_idx;
    synapse_results_fiber.input_cf = cf;
    synapse_results_fiber.input_freq = freq;
    synapse_results_fiber.input_db = db;
    synapse_results_fiber.input_toneidx = tone_idx;
    synapse_results_fiber.fiber_idx = fiber_idx_all;
    synapse_results_fiber.anf_type  = anf_type_all;
    synapse_results_fiber.psth      = psth_all;


    % === Clear temporary memory-heavy variables ===
    clear fiber_idx_all anf_type_all psth_all selected_fibers psth psth_raw spont tabs trel vihc
end
