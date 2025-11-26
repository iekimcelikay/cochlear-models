% we'll send a list of fibers and the IHC output (one per CF-tone chhanell) to a MATLAB helper function 
% 1. Loop over fibers 
% 2. call `model_synapse` per fiber 
% 3. Return all the outputs (or a struct array)

function synapse_results_fiber = run_fiber_synapse_batch(vihc, CF, nrep, dt, noiseType, implnt, expliketype)
	% Load fiber population struct
	fname = fullfile('~/PycharmProjects/subcorticalSTRF/subcorticalSTRF', 'ANpopulation.mat');
	fibers = load_AN_population_struct_by_type_then_cf(fname);
	nfibers = numel(fibers);  % <-- move this up
	cf_val = CF; 
	fs_raw = 1 / dt;
	fs_target = 1000; % Hz
	downsample_factor = round(fs_raw / fs_target);


    % Preallocate scalar struct with array fields
	synapse_results.fiber.cf = zeros(1,nfibers);
	synapse_results_fiber.cf_idx = zeros(1, nfibers);
    synapse_results_fiber.fiber_idx = zeros(1, nfibers);
    synapse_results_fiber.anf_type = cell(1, nfibers);
    synapse_results_fiber.psth = cell(1, nfibers);

	% Loop over fibers sequentially
	for idx_fiber = 1:nfibers 
		spont = fibers(idx_fiber).spont;
		tabs = fibers(idx_fiber).tabs;
		trel = fibers(idx_fiber).trel;

		% Run synapse model
		psth_raw = model_Synapse_BEZ2018a(vihc, CF, nrep, dt, noiseType, implnt, spont, tabs, trel, expliketype);

		% Resample PSTH
		if fs_target < fs_raw
			psth = decimate(double(psth_raw), downsample_factor);
		else
			psth = double(psth_raw);
		end

        % Store in scalar struct
		synapse_results.fiber.cf(idx_fiber) = cf_val;
		synapse_results_fiber.cf_idx(idx_fiber) = fibers(idx_fiber).cf_idx;
        synapse_results_fiber.fiber_idx(idx_fiber) = idx_fiber;
        synapse_results_fiber.anf_type{idx_fiber} = fibers(idx_fiber).anf_type;
        synapse_results_fiber.psth{idx_fiber} = psth;


		clear psth_raw psth;  % Help GC
	end
end


		
% parforloop