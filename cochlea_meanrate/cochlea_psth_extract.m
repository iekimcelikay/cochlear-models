function S = cochlea_psth_extract(outDir, defaultFs, targetFs, opts)
% Streaming / memory-light PSTH + mean-rate extractor.
% Usage:
%   S = cochlea_psth_extract(outDir, 1e5, 1000);
%   S = cochlea_psth_extract(outDir, 1e5, 1000, struct('storeRunPSTH',true));
%
% opts fields (all optional):
%   storeRunPSTH (false) : if true, saves each (CF,Tone,dB,Run,Type) PSTH to disk
%   psthDirName ('psth_runs') : subfolder under outDir for run PSTHs
%   verbose (true)
%
% Outputs:
%   cochlea_mean_rates.mat : mean_rates.(lsr|msr|hsr), sem_rates.(...), cfs, freqs, dbs, runs
%   If storeRunPSTH=true: individual PSTH files under psth_runs/*.mat
%
% NOTE: Does NOT assemble a giant psth_data cell array (memory saver).

if nargin < 1 || isempty(outDir)
    outDir = fullfile(fileparts(mfilename('fullpath')), 'cochlea_meanrate/out');
end
if nargin < 2 || isempty(defaultFs), defaultFs = 100000; end
if nargin < 3 || isempty(targetFs),  targetFs  = 1000;   end
if nargin < 4, opts = struct(); end
if ~isfield(opts,'storeRunPSTH'), opts.storeRunPSTH = false; end
if ~isfield(opts,'psthDirName'),  opts.psthDirName  = 'psth_runs'; end
if ~isfield(opts,'verbose'),      opts.verbose      = true; end

if exist('compute_psth_resampled','file') ~= 2
    error('compute_psth_resampled.m not on path.');
end

A = gather_cochlea_results(outDir, defaultFs);

anf_str   = lower(string(A.anf_type));
anf_types = ["lsr","msr","hsr"];

[cfs, ~, icf]    = unique(A.cf(:));
[freqs, ~, ifrq] = unique(A.tone_freq(:));
[dbs, ~, idb]    = unique(A.db(:));
[runs, ~, irun]  = unique(A.run(:));

numCF  = numel(cfs);
numF   = numel(freqs);
numDB  = numel(dbs);
numRun = numel(runs);

if opts.verbose
    fprintf('Streaming PSTH computation @ %g Hz (%d fibers total)\n', targetFs, numel(A.spikes));
end

% Allocate running stats (Welford) for mean firing rate per condition & fiber type
fiber_type_fields = {'lsr','msr','hsr'};
stats = struct();
for f = 1:numel(fiber_type_fields)
    stats.(fiber_type_fields{f}).n    = zeros(numCF, numF, numDB); % number of runs contributing
    stats.(fiber_type_fields{f}).mean = zeros(numCF, numF, numDB);
    stats.(fiber_type_fields{f}).M2   = zeros(numCF, numF, numDB); % sum of squared diffs
end

% Optional directory for individual PSTHs
psthRunDir = fullfile(outDir, opts.psthDirName);
if opts.storeRunPSTH
    if ~exist(psthRunDir,'dir'), mkdir(psthRunDir); end
end

% Main streaming loops
for cf_i = 1:numCF
    for frq_i = 1:numF
        for db_i = 1:numDB
            baseMask = (icf == cf_i) & (ifrq == frq_i) & (idb == db_i);
            for run_i = 1:numRun
                runMask = baseMask & (irun == run_i);

                for t = 1:numel(anf_types)
                    tmask = runMask & (anf_str == anf_types(t));
                    if ~any(tmask)
                        continue;
                    end

                    spkList = A.spikes(tmask);
                    fsList  = A.fs(tmask);

                    % Compute PSTH per fiber (resampled)
                    psths = cellfun(@(spk, fsrow) compute_psth_resampled(spk, fsrow, targetFs), ...
                                    spkList, num2cell(fsList), 'UniformOutput', false);

                    % Truncate to shortest length if needed
                    L = cellfun(@numel, psths);
                    Lmin = min(L);
                    if any(L ~= Lmin)
                        psths = cellfun(@(p) p(1:Lmin), psths, 'UniformOutput', false);
                    end

                    % Average across fibers
                    M = cell2mat(cellfun(@(x) x(:), psths, 'UniformOutput', false));
                    mean_psth = mean(M, 2);
                    mean_rate_this_run = mean(mean_psth); % scalar firing rate

                    field = char(anf_types(t));

                    % Welford update
                    n_old = stats.(field).n(cf_i, frq_i, db_i);
                    n_new = n_old + 1;
                    delta = mean_rate_this_run - stats.(field).mean(cf_i, frq_i, db_i);
                    mean_new = stats.(field).mean(cf_i, frq_i, db_i) + delta / n_new;
                    delta2 = mean_rate_this_run - mean_new;
                    M2_new = stats.(field).M2(cf_i, frq_i, db_i) + delta * delta2;

                    stats.(field).n(cf_i, frq_i, db_i)    = n_new;
                    stats.(field).mean(cf_i, frq_i, db_i) = mean_new;
                    stats.(field).M2(cf_i, frq_i, db_i)   = M2_new;

                    % Optional: save run PSTH
                    if opts.storeRunPSTH
                        fn = sprintf('PSTH_cf%d_freq%d_db%d_run%d_%s.mat', ...
                                     cf_i, frq_i, db_i, run_i, field);
                        save(fullfile(psthRunDir, fn), 'mean_psth', ...
                             'cf_i','frq_i','db_i','run_i','field','targetFs', ...
                             'mean_rate_this_run');
                    end
                end
            end
        end
    end
    if opts.verbose
        fprintf('Finished CF index %d / %d\n', cf_i, numCF);
    end
end

% Assemble mean & SEM
mean_rates = struct();
sem_rates  = struct();
for f = 1:numel(fiber_type_fields)
    field = fiber_type_fields{f};
    mean_rates.(field) = stats.(field).mean;
    sem_rates.(field)  = zeros(numCF, numF, numDB);
    nMat = stats.(field).n;
    M2Mat = stats.(field).M2;
    mask = nMat > 1;
    stdMat = zeros(size(nMat));
    stdMat(mask) = sqrt(M2Mat(mask) ./ (nMat(mask) - 1));
    sem_rates.(field)(mask) = stdMat(mask) ./ sqrt(nMat(mask));
    sem_rates.(field)(nMat == 1) = 0;          % single run -> SEM = 0
    mean_rates.(field)(nMat == 0) = NaN;       % no runs -> NaN
    sem_rates.(field)(nMat == 0)  = NaN;
end

outFileRates = fullfile(outDir, 'cochlea_mean_rates.mat');
save(outFileRates, 'mean_rates', 'sem_rates', 'cfs', 'freqs', 'dbs', 'runs', 'targetFs');
if opts.verbose
    fprintf('Saved mean rates: %s\n', outFileRates);
    if opts.storeRunPSTH
        fprintf('Per-run PSTHs stored under: %s\n', psthRunDir);
    end
end

% Return summary (no giant psth_data)
S = struct('mean_rates', mean_rates, 'sem_rates', sem_rates, ...
           'cfs', cfs, 'frequencies', freqs, 'dbs', dbs, 'runs', runs, ...
           'targetFs', targetFs, 'outFileRates', outFileRates, ...
           'psthRunDir', iff(opts.storeRunPSTH, psthRunDir, ''));

end

% Helper: inline ternary (since we removed big PSTH aggregator)
function y = iff(cond, a, b)
if cond, y = a; else, y = b; end
end

% --- Aggregation helper unchanged (kept for completeness) ---------------
function A = gather_cochlea_results(outDir, defaultFs)
pat = fullfile(outDir, 'cochlea_spike_rates_freq*_db*_cfbatch_*.mat');
files = dir(pat);
if isempty(files)
    pat2 = fullfile(outDir, 'cochlea_spike_rates_freq*_db*.mat');
    files = dir(pat2);
end
assert(~isempty(files), 'No cochlea_spike_rates_freq*.mat in %s', outDir);

cf = []; duration = []; tone_freq = []; db = []; run = [];
anf_type = {}; spikes = {}; cfb_start = []; cfb_end = []; fs_vec = [];
src = strings(0,1);

for k = 1:numel(files)
    fn = fullfile(files(k).folder, files(k).name);
    S = load(fn);
    at = S.anf_type;
    if isstring(at); at = cellstr(at);
    elseif ischar(at); at = cellstr(at);
    elseif isnumeric(at); at = cellstr(string(at));
    elseif ~iscell(at); at = cellfun(@char, at, 'uni', 0); end
    N = numel(S.cf);
    cf        = [cf; S.cf(:)];
    duration  = [duration; S.duration(:)];
    tone_freq = [tone_freq; S.tone_freq(:)];
    db        = [db; S.db(:)];
    run       = [run; S.run(:)];
    anf_type  = [anf_type; at(:)];
    spikes    = [spikes; S.spikes(:)];
    if isfield(S,'cf_batch_start'), cfb_start = [cfb_start; S.cf_batch_start(:)]; else, cfb_start = [cfb_start; nan(N,1)]; end
    if isfield(S,'cf_batch_end'),   cfb_end   = [cfb_end; S.cf_batch_end(:)];   else, cfb_end   = [cfb_end; nan(N,1)]; end
    fs_here = defaultFs;
    if isfield(S,'fs') && ~isempty(S.fs)
        if isscalar(S.fs)
            fs_here = S.fs;
        else
            try
                fs_here = S.fs(:);
                if numel(fs_here) ~= N, fs_here = repmat(defaultFs,N,1); end
            catch
                fs_here = repmat(defaultFs,N,1);
            end
        end
    end
    if isscalar(fs_here), fs_here = repmat(fs_here,N,1); end
    fs_vec = [fs_vec; fs_here(:)];
    src = [src; repmat(string(files(k).name), N, 1)];
end

spk_len = cellfun(@numel, spikes);
T = table(tone_freq, db, cf, run, string(anf_type), spk_len, src, ...
    'VariableNames', {'tone_freq','db','cf','run','anf_type','spk_len','src'});
[~, ia] = unique(T, 'rows', 'stable');

tone_freq = tone_freq(ia); db = db(ia); cf = cf(ia); duration = duration(ia);
run = run(ia); anf_type = anf_type(ia); cfb_start = cfb_start(ia); cfb_end = cfb_end(ia);
spikes = spikes(ia); fs_vec = fs_vec(ia); src = src(ia); spk_len = spk_len(ia);

[~, ord] = sortrows(table(tone_freq, db, cf, run, string(anf_type), spk_len));
tone_freq = tone_freq(ord); db = db(ord); cf = cf(ord); duration = duration(ord);
run = run(ord); anf_type = anf_type(ord); cfb_start = cfb_start(ord); cfb_end = cfb_end(ord);
spikes = spikes(ord); fs_vec = fs_vec(ord); src = src(ord);

A = struct('tone_freq', tone_freq, 'db', db, 'cf', cf, 'duration', duration, ...
           'anf_type', {anf_type}, 'run', run, 'cf_batch_start', cfb_start, ...
           'cf_batch_end', cfb_end, 'spikes', {spikes}, 'fs', fs_vec, ...
           'source_file', src, 'params', struct('note','Aggregated cochlea spikes','dir',outDir));
end