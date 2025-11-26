% Driver: process one CF at a time (low memory).
clear; clc;

base_dir  = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/cochlea_meanrate";
input_dir = fullfile(base_dir, "out");
targetFs  = 1000;
defaultFs = 100000;

% Manually list CFs to run (edit this):
cf_list = [1000 2000];  % <-- put desired CF values here

for k = 1:numel(cf_list)
    cf_val = cf_list(k);
    fprintf('\n=== Processing CF=%g (%d/%d) ===\n', cf_val, k, numel(cf_list));
    process_cochlea_cf(cf_val, input_dir, targetFs, defaultFs);
    % Remove everything except loop control + configuration.
    clearvars -except k cf_list base_dir input_dir targetFs defaultFs
end

fprintf('\nAll requested CFs done.\n');