import pandas as pd
import data_analysis.da1_data_quality.dataquality_class as class_dq
from utils.gtdf.generate_test_data import gdf  # get from here: https://github.com/dennisdeh/utils
from matplotlib import pyplot as plt

# %%
# 1: Load data
test = True  # if True,
if test:
    # generate test data
    df = gdf(n=1000, n_copies=3)
    feature_cols = ["int", "float", "str", "int_1", "float_1"]
    excluded_feature_cols = None
    date_col = "datetime"
    key_cols = ["str", "int_1"]
    missing_vals = {'str': ['A', 'B']}
    sample_freq = "M"
    date_start = None
    date_end = None
    file_template = None
else:
    df = pd.DataFrame()  # pass data frame.

dqa = class_dq.DQA(df=df,
                   feature_cols=None,
                   excluded_feature_cols=None,
                   date_col="datetime",
                   sample_freq='Y')
# %% Helper functions
dqa.helper_feature_types()
dqa.helper_missing_values()

# %% 1: Completeness
# dataframes
dqa.analyse_completeness(analyse_total=True)
print(dqa.df_missing)
print(dqa.df_completeness)
# plots
dqa.plot_completeness()
dqa.plots_completeness
dqa.plots_completeness["str"]
plt.show()
