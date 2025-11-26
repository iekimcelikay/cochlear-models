function fiber_table = load_AN_population_as_table(mat_filename)

% Load mat file
data = load(mat_filename);

% Flatten arrays and get counts
sponts_lsr = data.sponts_lsr(:);
tabss_lsr = data.tabss_lsr(:);
trels_lsr = data.trels_lsr(:);

n_lsr = numel(sponts_lsr);

sponts_msr = data.sponts_msr(:);
tabss_msr = data.tabss_msr(:);
trels_msr = data.trels_msr(:);

n_msr = numel(sponts_msr);

sponts_hsr = data.sponts_hsr(:);
tabss_hsr = data.tabss_hsr(:);
trels_hsr = data.trels_hsr(:);

n_hsr = numel(sponts_hsr);


% Concatenate all data 
all_sponts = [sponts_lsr; sponts_msr; sponts_hsr];
all_tabss = [tabss_lsr; tabss_msr; tabss_hsr];
all_trels = [trels_lsr; trels_msr; trels_hsr];

% Label types (keep as strings for now)
anf_type = [ ...
	repmat("lsr", n_lsr, 1);
	repmat("msr", n_msr, 1);
	repmat("hsr", n_hsr, 1)];

% Create fiber index
fiber_idx = (1:(n_lsr + n_msr + n_hsr))';

% Build the table (this is for my sanity so i can inspect it easier)
fiber_table = table(fiber_idx, anf_type, all_sponts, all_tabs, all_trels,...
		'VariableNames', {'fiber_idx', 'anf_type', 'spont', 'tabs', 'trel'});


end
