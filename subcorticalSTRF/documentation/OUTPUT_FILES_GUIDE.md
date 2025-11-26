# Output Files Guide

This guide explains the structure and usage of files saved to `model_comparison_PSTH_final/`.

## Directory Structure

```
model_comparison_PSTH_final/
├── Regression Plots (PNG, 300 DPI)
│   ├── regression_seaborn_cf_brightness_*dB_run*.png
│   ├── hsr_regression_cf_brightness_*dB_run*.png
│   ├── msr_regression_cf_brightness_*dB_run*.png
│   └── lsr_regression_cf_brightness_*dB_run*.png
│
├── Legends (PNG, 300 DPI)
│   ├── hsr_legend_*dB_run*.png
│   ├── msr_legend_*dB_run*.png
│   └── lsr_legend_*dB_run*.png
│
├── Vector Graphics (PDF)
│   └── *_seaborn_cf_brightness_*.pdf (if saved)
│
├── Correlation Comparison
│   └── correlation_comparison_*dB.png
│
└── Statistics for LaTeX
    ├── correlation_stats_*dB.json    (Machine-readable)
    ├── correlation_stats_*dB.txt     (Human-readable)
    └── correlation_table_*dB.tex     (LaTeX table)
```

## File Types

### 1. Regression Plots (PNG)
**Purpose**: Publication-quality scatter plots with regression lines  
**Resolution**: 300 DPI  
**Naming**: `{fiber}_regression_cf_brightness_{db}dB_run{run}.png`

**Visual Encoding**:
- **Color (hue)**: Characteristic Frequency (CF)
- **Brightness (alpha)**: Distance from CF (bright = at CF, dim = far from CF)
- Includes regression line, identity line, R², slope, p-value, and sample size

**Usage**:
- Insert directly into manuscripts
- Use in presentations
- Publication-ready quality

### 2. Legend Files (PNG)
**Purpose**: Separate legend with ALL CF values listed  
**Naming**: `{fiber}_legend_{db}dB_run{run}.png`

**Contents**:
- Complete list of all CF values with their colors
- Brightness encoding explanation
- Can be placed separately from main plot in multi-panel figures

### 3. Vector Graphics (PDF)
**Purpose**: Scalable vector graphics for journals  
**Naming**: Same as PNG but with `.pdf` extension

**Advantages**:
- Infinite zoom without quality loss
- Smaller file size for simple plots
- Preferred by many journals

### 4. Correlation Comparison Plots
**Purpose**: Side-by-side comparison of BEZ run-to-run vs Cochlea-BEZ correlations  
**Naming**: `correlation_comparison_{db}dB.png`

**Shows**:
- Box plots with scatter points
- BEZ run pairs (blue) vs Cochlea-BEZ (red)
- Mean correlations
- Standard deviations

### 5. Statistics Files

#### JSON Format (`correlation_stats_{db}dB.json`)
**Purpose**: Machine-readable data for further analysis

**Structure**:
```json
{
  "db_level": 60.0,
  "bez_run_pairwise": {
    "hsr": {
      "run_0_vs_run_1": {"r": 0.95, "p": 1e-50, "n": 400},
      ...
    },
    "hsr_summary": {"mean": 0.94, "std": 0.02, ...}
  },
  "cochlea_vs_bez": { ... },
  "summary": {
    "hsr": {
      "bez_mean_r": 0.94,
      "cochlea_mean_r": 0.87,
      "difference": 0.07
    }
  }
}
```

**Usage**:
- Load with `json.load()` in Python
- Import into R or MATLAB
- Automated analysis pipelines

#### Text Format (`correlation_stats_{db}dB.txt`)
**Purpose**: Human-readable summary

**Contains**:
- BEZ run pair-wise correlations (mean, std, min, max)
- Cochlea-BEZ correlations
- Comparison summary table

**Usage**:
- Quick reference
- Copy-paste into reports
- Appendix material

#### LaTeX Table (`correlation_table_{db}dB.tex`)
**Purpose**: Ready-to-use LaTeX table

**Example**:
```latex
\begin{table}[htbp]
  \centering
  \caption{Correlation statistics at 60 dB}
  \begin{tabular}{lccc}
    \hline
    Fiber Type & BEZ Run Pairs & Cochlea vs BEZ & Difference \\
    \hline
    HSR & $0.940 \pm 0.020$ & $0.870 \pm 0.015$ & $+0.070$ \\
    MSR & $0.920 \pm 0.025$ & $0.850 \pm 0.018$ & $+0.070$ \\
    LSR & $0.910 \pm 0.030$ & $0.830 \pm 0.020$ & $+0.080$ \\
    \hline
  \end{tabular}
  \label{tab:correlation_60dB}
\end{table}
```

**Usage**:
- Include directly in LaTeX documents: `\input{correlation_table_60dB.tex}`
- Customize caption and label as needed
- Modify table format (e.g., booktabs style)

## Workflow for LaTeX Integration

### 1. Copy Files to LaTeX Project
```bash
cp model_comparison_PSTH_final/correlation_table_60dB.tex manuscript/tables/
cp model_comparison_PSTH_final/*.png manuscript/figures/
```

### 2. Include in LaTeX Document
```latex
% In your .tex file
\section{Results}
We compared the consistency of BEZ runs with Cochlea-BEZ agreement (Table~\ref{tab:correlation_60dB}).

\input{tables/correlation_table_60dB.tex}

The regression analysis revealed strong correlations (Figure~\ref{fig:regression}).

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/hsr_regression_cf_brightness_60dB_run0.png}
  \caption{HSR fiber regression analysis at 60 dB.}
  \label{fig:regression}
\end{figure}
```

### 3. Extract Specific Values
```python
import json

with open('model_comparison_PSTH_final/correlation_stats_60dB.json', 'r') as f:
    stats = json.load(f)

# Use in LaTeX via Python
hsr_corr = stats['summary']['hsr']['bez_mean_r']
print(f"\\newcommand{{\\hsrCorr}}{{{hsr_corr:.3f}}}")
```

## Automated Multi-dB Analysis

To generate statistics for all dB levels:

```python
for db_level in [50, 60, 70, 80]:
    # Compute correlations...
    save_correlation_statistics(
        bez_pairwise, 
        cochlea_bez, 
        db_level=db_level,
        output_dir='model_comparison_PSTH_final'
    )
```

This creates:
- `correlation_stats_50dB.json/txt/tex`
- `correlation_stats_60dB.json/txt/tex`
- `correlation_stats_70dB.json/txt/tex`
- `correlation_stats_80dB.json/txt/tex`

## Tips

### For Presentations
- Use PNG plots at 300 DPI
- Include separate legend files for multi-panel slides
- Use correlation comparison plots to show key findings

### For Publications
- Use PDF vector graphics when possible
- Include LaTeX tables directly
- Reference JSON data in supplementary materials

### For Analysis
- Load JSON files for meta-analysis
- Compare statistics across dB levels
- Automate report generation

## File Naming Convention

```
{analysis_type}_{fiber}_{db}dB_run{run}.{ext}

Examples:
- regression_seaborn_cf_brightness_60dB_run0.png
- hsr_regression_cf_brightness_60dB_run0.png
- correlation_comparison_60dB.png
- correlation_stats_60dB.json
```

## Customization

To change the output directory:
```python
OUTPUT_DIR = 'my_custom_output_folder'
save_correlation_statistics(..., output_dir=OUTPUT_DIR)
```

To modify LaTeX table style:
```latex
% Use booktabs for professional tables
\usepackage{booktabs}

% Then modify the .tex file to use \toprule, \midrule, \bottomrule
% instead of \hline
```
