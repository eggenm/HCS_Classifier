"""
Running regression to explore how different parameterizations affect accuracy
across all analyses in a summary table.
"""
# =============================================================================
# Imports
# =============================================================================
import dirfuncs
import pandas as pd
import statsmodels.formula.api as smf
dropbox_dir = dirfuncs.guess_data_dir()


# =============================================================================
# Load data
# =============================================================================
data_dir = dropbox_dir
result_csv = data_dir + 'result.12252019.csv'
result_df = pd.read_csv(result_csv)

# =============================================================================
# Run regression
# =============================================================================
mod = smf.ols(formula="two_class_score_weighted ~ bands + class_scheme", data=result_df)
res = mod.fit()
print(res.summary())