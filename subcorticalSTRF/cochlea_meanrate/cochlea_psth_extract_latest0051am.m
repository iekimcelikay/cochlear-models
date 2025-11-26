%% Memory-efficient PSTH extraction
clear; clc;

input_dir = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate/out";
out_dir   = fullfile(input_dir, "condition_psths");
if ~exist(out_dir,'dir'), mkdir(out_dir); end

fs       = 100000;   % Simulation sampling rate
targetFs = 1000;     % PSTH target rate

file_list = dir(fullfile(input_dir,'*.mat'));

for kFile = 1:numel(file_list)
    fpath = fullfile(file_list(kFile).folder, file_list(kFile).name);
    S = load(fpath,'cf','spikes','anf_type','tone_freq','db');

    cf_per_fiber   = S.cf(:); % one CF per fiber
    unique_cf = unique(cf_per_fiber);

    % Normalize fiber type labels into a cell array of lowercase strings
    if iscell(S.anf_type)
        all_types = lower(strtrim(string(S.anf_type(:))));
    else
    % char matrix: convert rows to cellstr, then to string
        all_types = lower(strtrim(string(cellstr(S.anf_type))));
    end


    for icf = 1:numel(unique_cf)

        this_cf = unique_cf(icf);
        idxCF = (cf_per_fiber == this_cf);


        segSpikes = S.spikes(idxCF);
        labelsSegment = cellstr(all_types(idxCF));

        % Debug: print fiber type summary
        fprintf('CF=%g | Unique fiber types: ', this_cf);
        disp(unique(labelsSegment));

        % Initialize struct to hold averaged PSTHs
        psth_struct = struct('lsr',[],'msr',[],'hsr',[]);

        % Compute PSTH per fiber type
        fiber_types = {'lsr','msr','hsr'};

        for iType = 1:numel(fiber_types)
            ft = fiber_types{iType};  % char string
            idxType = strcmp(labelsSegment, ft);

            if ~any(idxType), continue; end  % skip if no fibers of this type

            theseSpikes = segSpikes(idxType); % cell array of spike times for this fiber type
            nFib = numel(theseSpikes);  % number of fibers of this type
            psth_sum = [];  % initialize sum of PSTHs
            fprintf('CF %.0f | %s: %d fibers\n', this_cf, ft, nFib)
            
            for iFiber = 1:nFib
                spk = theseSpikes{iFiber};

                [psthVec, ~] = compute_psth_resampled(spk, fs, targetFs);
                psthVec = psthVec(:); % enforce column vector
                if isempty(psth_sum)
                   psth_sum = psthVec;
                else
                   psth_sum = psth_sum + psthVec;
                end
            end
                psth_struct.(ft) = psth_sum / nFib;  % assign average
            
        end
        % Time axis from first non-empty PSTH
        N = 0;
        for ft = fiber_types
            if ~isempty(psth_struct.(ft{1}))
                N = length(psth_struct.(ft{1}));
                break;
            end
        end
        time = (0:N-1)'/targetFs;

        % Save PSTH struct
        save_name = sprintf('psth_cf%g_tone%g_db%g.mat', this_cf, S.tone_freq(1), S.db(1));
        save(fullfile(out_dir, save_name), 'time', 'psth_struct', '-v7.3');
    end
end

fprintf('Memory-efficient PSTH extraction done.\n');
