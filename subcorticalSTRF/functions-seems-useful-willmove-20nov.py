def create_cf_dataframe(bez_data, cochlea_data, fiber_type='lsr', db_idx=1):
    """
    Create a pandas DataFrame with BEZ vs Cochlea data for each CF
    
    This prepares the data in long format suitable for seaborn plotting functions.
    
    Args:
        bez_data: BEZ model data structure
        cochlea_data: Cochlea model data structure  
        fiber_type: Type of auditory nerve fiber ('lsr', 'msr', 'hsr')
        db_idx: Index for dB level (1 corresponds to 60 dB SPL)
        
    Returns:
        pandas.DataFrame: Data ready for seaborn regression plotting
    """
    
    # Extract parameters
    bez_cfs, bez_frequencies, bez_dbs = extract_parameters(bez_data, "bez")
    db_level = bez_dbs[db_idx]
    
    # Get firing rate data
    bez_rates = {
        'lsr': bez_data['bez_rates'].lsr,
        'msr': bez_data['bez_rates'].msr,
        'hsr': bez_data['bez_rates'].hsr
    }
    
    cochlea_rates = {
        'lsr': cochlea_data['lsr_all'],
        'msr': cochlea_data['msr_all'],
        'hsr': cochlea_data['hsr_all']
    }
    
    # Prepare data for DataFrame
    data_list = []
    
    print(f"Creating DataFrame for {fiber_type.upper()} fibers at {db_level} dB")
    
    for cf_idx, cf_val in enumerate(bez_cfs):
        print(f"Processing CF {cf_val:.0f} Hz...")
        
        for freq_idx, freq_val in enumerate(bez_frequencies):
            # Extract rates for this CF, frequency, and dB level
            bez_rate_raw = bez_rates[fiber_type][cf_idx, freq_idx, db_idx]
            cochlea_rate_raw = cochlea_rates[fiber_type][cf_idx, freq_idx, db_idx]
            
            # Handle individual time points (no averaging!)
            bez_points = np.atleast_1d(bez_rate_raw).flatten()
            cochlea_points = np.atleast_1d(cochlea_rate_raw).flatten()
            
            # Add each individual time point as separate data point
            for time_idx, (bez_val, cochlea_val) in enumerate(zip(bez_points, cochlea_points)):
                data_list.append({
                    'CF': float(cf_val),
                    'Frequency': float(freq_val),
                    'BEZ': float(bez_val),
                    'Cochlea': float(cochlea_val),
                    'Fiber_Type': fiber_type.upper(),
                    'dB_Level': float(db_level),
                    'Time_Point': time_idx
                })
    
    df = pd.DataFrame(data_list)
    print(f"DataFrame created with {len(df)} data points across {len(bez_cfs)} CFs")
    return df

def create_comparison_dataframe(bez_data, cochlea_data):
    """Create a combined dataframe for comparison plotting"""
    
    # Extract parameters
    bez_cfs, bez_frequencies, bez_dbs = extract_parameters(bez_data)
    cochlea_cfs, cochlea_frequencies, cochlea_dbs = extract_parameters(cochlea_data)
    
    # Extract PSTH data
    bez_rates = {
        'lsr': bez_data['bez_rates'].lsr,
        'msr': bez_data['bez_rates'].msr,
        'hsr': bez_data['bez_rates'].hsr
    }
    
    cochlea_rates = {
        'lsr': cochlea_data['lsr_all'],
        'msr': cochlea_data['msr_all'],
        'hsr': cochlea_data['hsr_all']
    }
    
    comparison_data = []
    
    print(f"Processing data dimensions: {len(bez_cfs)} CFs, {len(bez_frequencies)} frequencies, {len(bez_dbs)} dB levels")
    
    # Process each combination
    for cf_idx, cf_val in enumerate(bez_cfs):
        for freq_idx, freq_val in enumerate(bez_frequencies):
            for db_idx, db_val in enumerate(bez_dbs):
                try:
                    # Calculate frequency difference for brightness
                    freq_diff = freq_val - cf_val
                    
                    # Process each fiber type
                    for fiber_type in ['lsr', 'msr', 'hsr']:
                        # Get BEZ data
                        bez_psth = bez_rates[fiber_type][cf_idx, freq_idx, db_idx]
                        bez_mean = np.mean(bez_psth)
                        
                        # Get Cochlea data (using corrected indexing)
                        cochlea_psth = cochlea_rates[fiber_type][cf_idx, freq_idx, db_idx]
                        cochlea_mean = np.mean(cochlea_psth)
                        
                        comparison_data.append({
                            'cf': cf_val,
                            'frequency': freq_val,
                            'db_level': db_val,
                            'fiber_type': fiber_type,
                            'bez_rate': bez_mean,
                            'cochlea_rate': cochlea_mean,
                            'freq_diff': freq_diff,
                            'cf_group': f'CF_{cf_val:.0f}Hz'
                        })
                        
                except (IndexError, ValueError) as e:
                    continue
    
    df = pd.DataFrame(comparison_data)
    print(f"Created dataframe with {len(df)} comparison points")
    print(f"CF range: {df['cf'].min():.1f} - {df['cf'].max():.1f} Hz")
    print(f"Frequency difference range: {df['freq_diff'].min():.1f} - {df['freq_diff'].max():.1f} Hz")
    return df



def create_comparison_plots(df, db_level, output_dir):
    """Create seaborn scatter plots with CF as hue and frequency difference as brightness"""
    
    fiber_types = df['fiber_type'].unique()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get unique CFs for color mapping (same for all fiber types)
    unique_cfs = sorted(df['cf'].unique())
    n_cfs = len(unique_cfs)
    print(f"Found {n_cfs} characteristic frequencies (shared across all fiber types)")
    
    # Create perceptually uniform colors for CFs using seaborn (ONCE for all plots)
    # Use husl for perceptual uniformness across all CFs
    #cf_palette = sns.color_palette("husl", n_colors=n_cfs)
    cf_palette = sns.mpl_palette("Spectral", n_colors=n_cfs)  # Alternative palette
    cf_color_map = {cf: cf_palette[i] for i, cf in enumerate(unique_cfs)}
    
    for ft in fiber_types:
        ft_data = df[df['fiber_type'] == ft].copy()
        
        if len(ft_data) == 0:
            print(f"No data for {ft.upper()} fibers. Skipping.")
            continue
            
        print(f"Creating comparison plot for {ft.upper()} fibers ({len(ft_data)} points)...")
        
        # Normalize frequency difference for brightness (alpha values)
        freq_diff_range = ft_data['freq_diff'].max() - ft_data['freq_diff'].min()
        print(f"  Frequency difference range: {ft_data['freq_diff'].min():.1f} to {ft_data['freq_diff'].max():.1f} Hz")
        
        # Create figure with seaborn styling
        plt.figure(figsize=(14, 10))
        
        # Calculate alpha values based on absolute frequency difference
        # Closer to CF = brighter (higher alpha), farther = dimmer (lower alpha)
        abs_freq_diff = np.abs(ft_data['freq_diff'])
        max_abs_diff = abs_freq_diff.max()
        
        # Make brightness differences more dramatic (0.1 to 1.0 range)
        # Use exponential scaling for more pronounced differences
        normalized_diff = abs_freq_diff / max_abs_diff
        ft_data['alpha_val'] = 0.1 + 0.9 * (1 - normalized_diff)**2  # Exponential curve for more contrast
        
        print(f"  Alpha range: {ft_data['alpha_val'].min():.2f} to {ft_data['alpha_val'].max():.2f}")
        print(f"  Sample alphas - Near CF: {ft_data['alpha_val'].max():.2f}, Far from CF: {ft_data['alpha_val'].min():.2f}")
        
        # Create scatter plot for each CF with more pronounced brightness
        for cf_idx, cf_val in enumerate(unique_cfs):
            cf_data = ft_data[ft_data['cf'] == cf_val].copy()
            
            if len(cf_data) == 0:
                continue
            
            # Plot points with enhanced visibility - larger size for better alpha perception
            # Plot points with enhanced visibility - larger size for better alpha perception
            for _, point in cf_data.iterrows():
                plt.scatter(point['bez_rate'], point['cochlea_rate'], 
                           c=[cf_color_map[cf_val]], alpha=point['alpha_val'], 
                           s=80, edgecolors='black', linewidths=0.5,  # Larger points with thicker edges
                           label=f'CF {cf_val:.0f}Hz' if _ == cf_data.index[0] else "")
        
        # Data quality diagnostics before regression
        print(f"  Data quality check:")
        print(f"    BEZ rates - Min: {ft_data['bez_rate'].min():.3f}, Max: {ft_data['bez_rate'].max():.3f}")
        print(f"    Cochlea rates - Min: {ft_data['cochlea_rate'].min():.3f}, Max: {ft_data['cochlea_rate'].max():.3f}")
        print(f"    NaN values - BEZ: {ft_data['bez_rate'].isna().sum()}, Cochlea: {ft_data['cochlea_rate'].isna().sum()}")
        print(f"    Infinite values - BEZ: {np.isinf(ft_data['bez_rate']).sum()}, Cochlea: {np.isinf(ft_data['cochlea_rate']).sum()}")
        
        # Clean data for regression (remove NaN and infinite values)
        clean_mask = (
            np.isfinite(ft_data['bez_rate']) & 
            np.isfinite(ft_data['cochlea_rate']) &
            ~ft_data['bez_rate'].isna() &
            ~ft_data['cochlea_rate'].isna()
        )
        
        clean_data = ft_data[clean_mask].copy()
        if len(clean_data) != len(ft_data):
            print(f"    Removed {len(ft_data) - len(clean_data)} problematic points")
        
        # Calculate statistics using cleaned data
        if len(clean_data) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(clean_data['bez_rate'], clean_data['cochlea_rate'])
            
            # Add regression line manually (like MATLAB's polyfit approach)
            x_range = np.linspace(clean_data['bez_rate'].min(), clean_data['bez_rate'].max(), 100)
            y_fit = slope * x_range + intercept
            plt.plot(x_range, y_fit, 'k-', linewidth=2, alpha=0.8, label='Linear fit')
            
            # Verify with numpy polyfit (MATLAB equivalent)
            poly_coeffs = np.polyfit(clean_data['bez_rate'], clean_data['cochlea_rate'], 1)
            matlab_slope, matlab_intercept = poly_coeffs[0], poly_coeffs[1]
            
            # Calculate R² manually (like MATLAB's corrcoef)
            correlation_matrix = np.corrcoef(clean_data['bez_rate'], clean_data['cochlea_rate'])
            matlab_r_squared = correlation_matrix[0, 1]**2
            
            print(f"  Linear regression comparison:")
            print(f"    scipy.linregress - Slope: {slope:.6f}, R²: {r_value**2:.6f}")
            print(f"    numpy.polyfit   - Slope: {matlab_slope:.6f}, R²: {matlab_r_squared:.6f}")
            print(f"    Difference      - Slope: {abs(slope - matlab_slope):.6f}, R²: {abs(r_value**2 - matlab_r_squared):.6f}")
        else:
            slope, intercept, r_value, p_value = 0, 0, 0, 1
            print(f"  Not enough clean data points for regression")
        
        # Add diagonal reference line (perfect match)
        min_val = min(ft_data['bez_rate'].min(), ft_data['cochlea_rate'].min())
        max_val = max(ft_data['bez_rate'].max(), ft_data['cochlea_rate'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect match')
        
        # Add statistics text with both methods
        stats_text = f'R² = {r_value**2:.3f}\nSlope = {slope:.3f}\nIntercept = {intercept:.3f}\nn = {len(clean_data)}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        print(f"  Final stats - Slope: {slope:.3f}, R²: {r_value**2:.3f}, n: {len(clean_data)}")
        
        # Formatting with seaborn aesthetics
        plt.xlabel('BEZ model firing rate (spikes/s)', fontsize=12, fontweight='bold')
        plt.ylabel('Cochlea model firing rate (spikes/s)', fontsize=12, fontweight='bold')
        plt.title(f'{ft.upper()} fibers at {int(db_level)} dB: BEZ vs. Cochlea\n' +
                 f'Color = Characteristic Frequency (perceptually uniform), Brightness = Distance from CF', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Create custom legend showing ALL CFs with perceptually uniform colors
        legend_entries = []
        for i, cf_val in enumerate(unique_cfs):  # Show all CFs
            legend_entries.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=cf_color_map[cf_val], markersize=6,
                                           markeredgecolor='white', markeredgewidth=0.5,
                                           label=f'CF {cf_val:.0f}Hz'))
        
        # Add brightness explanation with better visibility
        legend_entries.extend([
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', alpha=0.1, markersize=8,
                      markeredgecolor='black', markeredgewidth=0.5,
                      label='Far from CF (dim)'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', alpha=1.0, markersize=8,
                      markeredgecolor='black', markeredgewidth=0.5,
                      label='At CF (bright)')
        ])
        
        # Style the legend
        legend = plt.legend(handles=legend_entries, bbox_to_anchor=(1.02, 0.5), 
                           loc='center left', fontsize=8, frameon=True, 
                           fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        
        # Apply seaborn styling to the plot
        sns.despine()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(output_dir, f'compare_{ft}_cf_brightness_seaborn_db{int(db_level)}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        
        print(f"  Plot saved to {output_file}")
        
        # Keep plot open without blocking
        plt.show(block=False)


