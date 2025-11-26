function synapse_df = load_synapse_df_matlab(mat_filename):
	data = load(mat_filename);

	[numcfs, num_lsr] = size(data.sponts_lsr);
	[~, num_msr] = size(data.sponts_msr);
	[~, num_hsr] = size(data.sponts_hsr);


	total_fibers = numcfs * (num_lsr + num_msr + num_hsr);
	fiber_struct = repmat(struct(), total_fibers, 1);
	idx = 1; 

	% === LSR fibers ===

	for j = 1:num_lsr
		for cf_idx = 1:numcfs
			fiber_struct(idx).spont = data.sponts_lsr(cf_idx, j);
			fiber_struct(idx).tabs = data.tabss_lsr(cf_idx, j);
			fiber_struct(idx).trel = data.trels_lsr(cf_idx, j);
			fiber_struct(idx).anf_type = 'lsr';
			fiber_struct(idx).cf_idx = cf_idx;
			idx = idx + 1;
		end
	end

	% === MSR fibers ===

	for j = 1:num_msr
		for cf_idx = 1:numcfs
			fiber_struct(idx).spont = data.sponts_msr(cf_idx, j);
			fiber_struct(idx).tabs = data.tabss_msr(cf_idx, j);
			fiber_struct(idx).trel = data.trels_msr(cf_idx, j);
			fiber_struct(idx).anf_type = 'msr';
			fiber_struct(idx).cf_idx = cf_idx;
			idx = idx + 1;
		end
	end

	% === HSR fibers ===

	for j = 1:num_hsr
		for cf_idx = 1:numcfs
			fiber_struct(idx).spont = data.sponts_hsr(cf_idx, j);
			fiber_struct(idx).tabs = data.tabss_hsr(cf_idx, j);
			fiber_struct(idx).trel = data.trels_hsr(cf_idx, j);
			fiber_struct(idx).anf_type = 'hsr';
			fiber_struct(idx).cf_idx = cf_idx;
			idx = idx + 1;
		end
	end
end
