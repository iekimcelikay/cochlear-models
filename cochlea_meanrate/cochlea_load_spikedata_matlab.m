% Load spike_data from python

load('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate/all_spike_data.mat');
anf_type = cellstr(anf_type);  % Now a 180×1 cell array of 'hsr', 'msr', etc.

if ~isa(cf, 'double')
    cf = double(cf);
end

if ~isa(tone_freq, 'double')
    tone_freq = double(tone_freq);
end

if ~isa(db, 'double')
    db = double(db);
end

% Optionally ensure each cell contains a row vector:
spikes = cellfun(@(x) double(x(:).'), spikes, 'UniformOutput', false);

T = table(cf, anf_type, tone_freq, db, spikes);
sorted_T = sortrows(T, "cf")
