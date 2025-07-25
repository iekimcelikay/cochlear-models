function fiber_struct = load_AN_population_struct_by_type_then_cf(mat_filename)
	data = load(mat_filename);

	[numcfs, num_lsr] = size(data.sponts_lsr);
	[~, num_msr] = size(data.sponts_msr);
	[~, num_hsr] = size(data.sponts_hsr);


	total_fibers = numcfs * (num_lsr + num_msr + num_hsr);
	fiber_struct = repmat(struct(), total_fibers, 1);
	idx = 1; 

	for cf_idx = 1:numcfs
		local_fiber_idx = 1;
		% LSR fibers for this CF
		for j = 1:num_lsr
            fiber_struct(idx).spont = data.sponts_lsr(cf_idx, j);
            fiber_struct(idx).tabs = data.tabss_lsr(cf_idx, j);
            fiber_struct(idx).trel = data.trels_lsr(cf_idx, j);
            fiber_struct(idx).anf_type = 'lsr';
            fiber_struct(idx).cf_idx = cf_idx;
            fiber_struct(idx).fiber_idx = local_fiber_idx;
            fiber_struct(idx).local_idx = j;
            idx = idx + 1;
			local_fiber_idx = local_fiber_idx + 1;
		end

		% MSR Fibers for this cf
		for j = 1:num_msr
			fiber_struct(idx).spont = data.sponts_msr(cf_idx, j);
			fiber_struct(idx).tabs = data.tabss_msr(cf_idx, j);
			fiber_struct(idx).trel = data.trels_msr(cf_idx, j);
			fiber_struct(idx).anf_type = 'msr';
			fiber_struct(idx).cf_idx = cf_idx;
            fiber_struct(idx).fiber_idx = local_fiber_idx;
            fiber_struct(idx).local_idx = j;
            idx = idx + 1;
			local_fiber_idx = local_fiber_idx + 1;
		end

		 % HSR fibers for this CF
		for j = 1:num_hsr
            fiber_struct(idx).spont = data.sponts_hsr(cf_idx, j);
            fiber_struct(idx).tabs = data.tabss_hsr(cf_idx, j);
            fiber_struct(idx).trel = data.trels_hsr(cf_idx, j);
            fiber_struct(idx).anf_type = 'hsr';
            fiber_struct(idx).cf_idx = cf_idx;
            fiber_struct(idx).fiber_idx = local_fiber_idx;
            fiber_struct(idx).local_idx = j;
            idx = idx + 1;
			local_fiber_idx = local_fiber_idx + 1;
		end
	end
end
