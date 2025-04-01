# File: analyze_resonance_enhanced_stdp.py
# Purpose: Analyze simulation results from resonance_enhanced_stdp.py to validate selective modulation (Section 4.L.2)
# Dependencies: Assumes resonance_enhanced_stdp.py has been run and results are saved in simulation_results.npy

import numpy as np
from typing import List, Tuple
import os # Import os to check for file existence

# Define the expected structure of each result tuple for clarity
# (PLV, correlation, avg_weight_change_correlated_group, 
#  avg_weight_change_uncorrelated_group, selectivity_ratio, final_weights)
# Note: The actual saved data might have slightly different indexing depending on the exact tuple structure saved.
# Adjust indices in the analysis function if needed.

def analyze_simulation_results(results: np.ndarray) -> Tuple[bool, str]:
    """
    Analyze simulation results loaded from file to confirm the selective modulation 
    principle of Resonance-Enhanced STDP.
    
    Args:
        results (np.ndarray): Numpy array loaded from file, where each row is a result tuple.
                              Expected structure per row: 
                              [plv, correlation, avg_corr, avg_uncorr, selectivity, weights_array]
    
    Returns:
        Tuple[bool, str]: (is_validated, validation_summary)
    """
    if results.ndim == 0 or results.size == 0:
         return False, "Analysis failed: Loaded results array is empty or invalid."
         
    # Assuming the structure saved is:
    # Index 0: PLV
    # Index 1: Correlation
    # Index 2: Avg Abs Weight Change (Diagonal/Correlated)
    # Index 3: Avg Abs Weight Change (Off-Diagonal/Uncorrelated)
    # Index 4: Selectivity Ratio
    # Index 5: Final Weights (np.ndarray)

    try:
        # Check 1: Representativeness of Correlation (using correlation levels directly)
        # Expected: Higher correlation levels should generally lead to higher selectivity ratios, 
        #           especially at higher PLV levels.
        correlation_levels = sorted(list(set(results[:, 1]))) # Unique correlation levels tested
        plv_levels = sorted(list(set(results[:, 0])))         # Unique PLV levels tested
        
        correlation_summary_lines = ["Correlation Representativeness Check:"]
        correlation_trend_valid = True # Assume valid initially
        for plv in plv_levels:
            line = f"  PLV={plv:.2f}: "
            selectivities_at_plv = []
            for corr in correlation_levels:
                 # Find results matching current plv and corr
                 mask = (results[:, 0] == plv) & (results[:, 1] == corr)
                 if np.any(mask):
                      selectivity = results[mask, 4][0] # Get selectivity ratio
                      selectivities_at_plv.append(selectivity)
                      line += f"Corr={corr:.2f}(Sel={selectivity:.3f}) "
                 else:
                      selectivities_at_plv.append(np.nan) # Mark missing data point
                      line += f"Corr={corr:.2f}(N/A) "
            
            # Check if selectivity generally increases with correlation at this PLV
            valid_selectivities = [s for s in selectivities_at_plv if not np.isnan(s)]
            if len(valid_selectivities) > 1:
                 # Simple check: is the list mostly increasing?
                 increases = sum(1 for i in range(len(valid_selectivities) - 1) if valid_selectivities[i+1] > valid_selectivities[i])
                 if increases < (len(valid_selectivities) - 1) * 0.6: # Allow for some noise/non-monotonicity
                      correlation_trend_valid = False
                      line += " [Trend Invalid]"
            correlation_summary_lines.append(line)
        correlation_summary = "\n".join(correlation_summary_lines) + f"\nOverall Trend Valid: {correlation_trend_valid}\n"


        # Check 2: Robustness of Selectivity Metric (using selectivity ratio directly)
        # Expected: Higher PLV should lead to higher selectivity ratios, especially at higher correlations.
        selectivity_summary_lines = ["Selectivity Metric Robustness Check:"]
        selectivity_trend_valid = True # Assume valid initially
        for corr in correlation_levels:
             line = f"  Correlation={corr:.2f}: "
             selectivities_at_corr = []
             for plv in plv_levels:
                  mask = (results[:, 0] == plv) & (results[:, 1] == corr)
                  if np.any(mask):
                       selectivity = results[mask, 4][0]
                       selectivities_at_corr.append(selectivity)
                       line += f"PLV={plv:.2f}(Sel={selectivity:.3f}) "
                  else:
                       selectivities_at_corr.append(np.nan)
                       line += f"PLV={plv:.2f}(N/A) "

             valid_selectivities = [s for s in selectivities_at_corr if not np.isnan(s)]
             if len(valid_selectivities) > 1:
                  increases = sum(1 for i in range(len(valid_selectivities) - 1) if valid_selectivities[i+1] > valid_selectivities[i])
                  if increases < (len(valid_selectivities) - 1) * 0.6:
                       selectivity_trend_valid = False
                       line += " [Trend Invalid]"
             selectivity_summary_lines.append(line)
        selectivity_summary = "\n".join(selectivity_summary_lines) + f"\nOverall Trend Valid: {selectivity_trend_valid}\n"


        # Check 3: Parameter Space Exploration (Implicitly covered by above checks across PLV/Corr)
        # We check if selectivity is consistently above a threshold across conditions.
        min_selectivity_threshold = 1.1 # Define a minimum meaningful selectivity ratio
        all_selectivities = results[:, 4].astype(float)
        param_selectivity_valid = np.all(all_selectivities[~np.isnan(all_selectivities)] > min_selectivity_threshold)
        min_observed_selectivity = np.min(all_selectivities[~np.isnan(all_selectivities)]) if len(all_selectivities[~np.isnan(all_selectivities)]) > 0 else np.nan
        
        param_summary = (f"Parameter Space Check:\n"
                         f"  Minimum Selectivity Ratio Observed: {min_observed_selectivity:.4f}\n"
                         f"  Required Minimum Selectivity Ratio: {min_selectivity_threshold}\n"
                         f"  Selectivity Consistently Above Threshold: {param_selectivity_valid}\n")

        # Overall Validation Decision
        is_validated = correlation_trend_valid and selectivity_trend_valid and param_selectivity_valid
        validation_summary = (f"--- Validation Summary for Resonance-Enhanced STDP (Section 4.L.2) ---\n"
                              f"{correlation_summary}"
                              f"{selectivity_summary}"
                              f"{param_summary}"
                              f"--------------------------------------------------------------------\n"
                              f"Proceed to Basic Resonance Measurement [Sec 4.L.1]: {is_validated}\n"
                              f"--------------------------------------------------------------------")
        
        return is_validated, validation_summary
        
    except IndexError as e:
         return False, f"Analysis failed: IndexError likely due to unexpected results structure. Error: {e}"
    except Exception as e:
        print(f"Analysis failed during processing: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Analysis failed: {e}"

# Run the analysis 
if __name__ == "__main__":
    RESULTS_FILENAME = 'simulation_results.npy'
    
    if not os.path.exists(RESULTS_FILENAME):
        print(f"Error: Results file not found at '{RESULTS_FILENAME}'.")
        print("Please run 'resonance_enhanced_stdp.py' first to generate the results.")
    else:
        try:
            # Load results from the file
            # allow_pickle=True is necessary because the array contains numpy arrays (weights)
            loaded_results = np.load(RESULTS_FILENAME, allow_pickle=True)
            print(f"Successfully loaded results from {RESULTS_FILENAME}")
            
            # Analyze the loaded results
            is_validated, summary = analyze_simulation_results(loaded_results)
            print("\n" + summary)
            
        except Exception as e:
            print(f"Error loading or analyzing results file '{RESULTS_FILENAME}': {e}")
            import traceback
            traceback.print_exc()
