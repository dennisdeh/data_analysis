import datetime
import os
import pandas as pd
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
import yaml


class DQA:
    """
    Data quality analysis (DQA) class.

    Can be used with many different data sources and gives assessment
    of quality metrics defined through a threshold file. It can run with
    few parameters, but there is the possibility to change the default
    behavior to a large extent.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Union[str, list, None] = None,
        excluded_feature_cols: Union[str, list, None] = None,
        date_col: Union[str, None] = None,
        key_cols: Union[str, list, None] = None,
        force_missing_values: Union[dict, None] = None,
        force_datatypes: Union[dict, None] = None,
        sample_freq: Union[str, None] = "M",
        date_start: Union[str, None] = None,
        date_end: Union[str, None] = None,
        keep_nat: bool = True,
        verify_keys: bool = True,
        path_template: Union[str, bool, str] = False,
        path_thresholds: Union[str, None] = None,
    ):
        """
        Initialize the DQA class.

        Parameters
        ---------
        df : pd.DataFrame
            data frame to be analysed.
        feature_cols : str, list, (optional)
            list of columns to analyse, by default all are selected
        excluded_feature_cols : str, list, (optional)
            list of columns to exclude from analysis, by default none are excluded
        date_col : str (optional)
            date column name, to be provided if time-dependent analyses are to be
            carried out.
        key_cols : str, list, (optional)
            the key columns to use. If None, the standard pandas index defined
            will be used in the analyses.
        force_missing_values: dict (optional)
            dictionary with column names as keys, with values as a list specifying
            which values of the feature that are to be considered missing.
        force_datatypes: dict (optional)
            dictionary with column names as keys, with values as a list specifying
            the data type of the corresponding column, i.e. "num" (if numerical),
            "datetime" (if representing time and/or date), "cat" (if categorical).
        sample_freq: str (optional)
            (re)sampling frequency: can be D, M Y or None
        date_start: str (optional)
            start date of analysis. If not given, the entire available
            data range will be used.
        date_end: str (optional)
            end date of analysis. If not given, the entire available
            data range will be used.
        keep_nat: bool (optional)
            whether to keep not-a-time values in date_col. Per default, they are
            discarded.
        verify_keys: bool (optional)
            whether to check consistency of the index keys provided.
        path_template: bool, str (optional)
            path to Excel template file for report outputs. If None,
            the file dq_template.xlsm in the working directory is used.
        path_template: bool, str (optional)
            path to Excel template file for report outputs. If None,
            the file dq_template.xlsm in the working directory is used.
        """
        # step 1: Set class attributes
        self.df = df.copy(deep=True)
        self.feature_cols = feature_cols
        self.excluded_feature_cols = excluded_feature_cols
        self.date_col = date_col
        self.key_cols = key_cols
        self.force_missing_values = force_missing_values
        self.force_datatypes = force_datatypes
        self.sample_freq = sample_freq
        self.date_start = date_start
        self.date_end = date_end
        self.keep_NaT = keep_nat
        self.verify_keys = verify_keys
        if path_template is None or path_template:
            self.path_template = os.path.join(os.getcwd(), "dq_template.xlsm")
        elif isinstance(path_template, str):
            self.path_template = path_template
        elif not path_template:
            # no template used; no report can be saved
            self.path_template = False
        else:
            raise FileNotFoundError("invalid input for path_template")
        assert not self.path_template or os.path.isfile(self.path_template), \
            f"Could not find template file at: {self.path_template}"
        if path_thresholds is None:
            self.path_thresholds = os.path.join(os.getcwd(), "dq_thresholds.yml")
        elif isinstance(path_thresholds, str):
            self.path_thresholds = path_thresholds
        else:
            raise FileNotFoundError("invalid input for path_thresholds")
        assert os.path.isfile(self.path_thresholds), \
            f"Could not find thresholds file at: {self.path_thresholds}"

        # step 2: Initialise other structures
        # initialise dataframes
        self.df_missing = pd.DataFrame()
        self.df_completeness = pd.DataFrame()
        self.df_thresholds = pd.DataFrame()
        # data types
        self.datatypes_numeric = [
            "float",
            "float64",
            "float32",
            "Int64",
            "int",
            "int64",
        ]
        self.datatypes_categorical = ["object", "category", "bool"]
        self.datatypes_datetime = ["datetime64[ns]", "timedelta[ns]"]
        # overall lists and dicts
        self.excluded_cols = []
        self.dict_datatypes = {}
        self.dict_thresholds = {}
        # dicts for plots
        self.plots_completeness = {}
        # flags
        self.flag1 = False

        # step 3: Verify and convert structure
        l0 = len(self.df)
        # step 3.1: Features - Get feature columns names, assert columns are present
        print("Loading data and verifying structure:")
        if self.feature_cols is None:
            print("Analysing all available columns")
            self.feature_cols = self.df.columns
        elif isinstance(self.feature_cols, str):
            self.feature_cols = [self.feature_cols]
            assert (
                pd.Index(self.feature_cols).isin(self.df.columns)
            ).all(), "Not all columns in feature_cols are present in df."
        elif isinstance(self.feature_cols, list):
            print(f"Analysing the {len(self.feature_cols)} given columns")
            self.feature_cols = list(set(self.feature_cols))  # remove duplicates
            assert (
                pd.Index(self.feature_cols).isin(self.df.columns)
            ).all(), "Not all columns in feature_cols are present in df."
        else:
            raise TypeError("Invalid type of input for feature_cols.")

        # step 3.2: Dates - Get date column if it exists
        if self.date_col is None:
            print("No date column given, performing time-independent analyses")
        elif isinstance(self.date_col, str):
            assert self.date_col in df.columns, "date_col is not a column of df."
            print("Date column given, performing time-dependent analyses")
        else:
            raise TypeError("Invalid type of input for date_col.")

        # step 3.3: Key columns
        print("Key columns:")
        # step 3.3.1: Initialise key columns
        if self.key_cols is None or len(self.key_cols) == 0:
            self.key_cols = []
            print("   No key columns given, using row index as key.")
        elif isinstance(self.key_cols, str):
            print(f"   Key column is: {self.key_cols}.")
            self.key_cols = [self.key_cols]
            assert self.key_cols in df.columns, "key col is not a column of df."
            if self.verify_keys:
                assert not self.df.set_index(
                    self.key_cols
                ).index.has_duplicates, (
                    "key column is not consistent, it has duplicates."
                )
        else:
            print(f"   Key columns are: {', '.join(self.key_cols)}.")
            assert (
                pd.Index(self.key_cols).isin(self.df.columns)
            ).all(), "key_cols are not columns of df."
            if self.verify_keys:
                assert not self.df.set_index(
                    self.key_cols
                ).index.has_duplicates, (
                    "key columns are not consistent, they have duplicates."
                )

        # step 3.3.2: Remove key columns and exclude those not of interest (date etc.)
        if self.excluded_feature_cols is None:
            self.excluded_feature_cols = []
            if self.date_col is None:
                self.excluded_cols = list(set(self.key_cols))
            else:
                self.excluded_cols = list(set(self.key_cols + [self.date_col]))
            self.feature_cols = list(
                set(self.feature_cols).difference(self.excluded_cols)
            )
        elif isinstance(self.excluded_feature_cols, str):
            self.excluded_feature_cols = [self.excluded_feature_cols]
            if self.date_col is None:
                self.excluded_cols = list(
                    set(self.excluded_feature_cols).union(set(self.key_cols))
                )
            else:
                self.excluded_cols = list(
                    set(self.excluded_feature_cols).union(
                        set(self.key_cols + [self.date_col])
                    )
                )
            self.feature_cols = list(
                set(self.feature_cols).difference(self.excluded_cols)
            )
            assert (
                pd.Index(self.excluded_feature_cols).isin(self.df.columns)
            ).all(), "excluded feature is not a column of df."
        elif isinstance(self.excluded_feature_cols, list):
            self.excluded_feature_cols = list(
                set(self.excluded_feature_cols)
            )  # remove duplicates
            if self.date_col is None:
                self.excluded_cols = list(
                    set(self.excluded_feature_cols).union(set(self.key_cols))
                )
            else:
                self.excluded_cols = list(
                    set(self.excluded_feature_cols).union(
                        set(self.key_cols + [self.date_col])
                    )
                )
            self.feature_cols = list(
                set(self.feature_cols).difference(self.excluded_cols)
            )
            assert (
                pd.Index(self.excluded_feature_cols).isin(self.df.columns)
            ).all(), "excluded_feature_cols are not columns of df."
        else:
            raise TypeError("Invalid input for excluded_feature_cols.")

        # step 3.3.3: Check that excluded columns do not contain key columns.
        if len(self.excluded_feature_cols) != 0:
            assert set(self.excluded_feature_cols).isdisjoint(
                set(self.key_cols)
            ), "The excluded columns contains one or more key columns."
            print(
                f"Columns that are excluded from analysis: {', '.join(self.excluded_feature_cols)}."
            )

        # step 3.3.4: Check that date column is not in excluded columns
        str_datecol = "0 date columns"
        if isinstance(self.date_col, str):
            assert set(self.excluded_feature_cols).isdisjoint(
                {self.date_col}
            ), "Date column is in the list of excluded columns."
            if not set(self.key_cols).isdisjoint({self.date_col}):
                print("Note: Date column is also a key column.")
            self.feature_cols = list(set(self.feature_cols).difference({self.date_col}))
            str_datecol = "1 date column"

        # step 3.4: Print summary
        print(
            f"Summary of structure:\n   {l0} rows in df\n   {len(self.df.columns)} columns in df"
            f"\n   {len(self.feature_cols)} feature columns"
            f"\n   {len(self.excluded_feature_cols)} excluded feature columns"
            f"\n   {len(self.key_cols)} key columns"
            f"\n   {str_datecol}"
        )

        # step 4: filter columns of df so that we only keep what is necessary, convert date_col
        if self.date_col is None:
            self.df = self.df[self.key_cols + self.feature_cols]
        else:
            self.df = self.df[
                list(set(self.key_cols + [self.date_col])) + self.feature_cols
            ]  # remove duplicates
            self.df[date_col] = pd.to_datetime(self.df[date_col])

        # step 5: Restrict date range if needed
        if not (self.date_col is None):
            print("Filtering of dates:")
            # start date
            if self.date_start is None:
                print("   date_start: None")
                # if self.date_start is None:  # set the earliest data date if nothing else is given
                self.date_start = min(self.df[self.date_col])
            elif isinstance(self.date_start, str):
                self.date_start = datetime.datetime.strptime(
                    self.date_start, "%Y-%m-%d"
                )
                print(f"   date_start: {str(self.date_start)}")
            # end date
            if self.date_end is None:
                print("   date_end: None")
                # if self.date_end is None:  # set the latest data date if nothing else is given
                self.date_end = max(self.df[self.date_col])
            elif isinstance(self.date_end, str):
                self.date_end = datetime.datetime.strptime(self.date_end, "%Y-%m-%d")
                print(f"   date_end: {str(self.date_end)}")
            # filter data frame
            if not self.keep_NaT:
                print("   NaT's are also filtered out")
            mask = (
                (self.df[date_col] >= self.date_start)
                & (  # Filter dates for plotting range; make boolean mask
                    self.df[date_col] <= self.date_end
                )
            ) | (self.keep_NaT & self.df[date_col].isna())
            self.df = self.df.loc[mask]
            l1 = len(self.df)
            print(f"   observations filtered out (incl. NaTs): {l0 - l1}")
            print(f"   new number of rows: {l1}")

        # step 6: Missing values encoding
        # step 6.1: general conversion of null, 'nan' etc.
        self.df = self.df.replace(
            to_replace=["null", "na", "nan", None], value=np.nan
        ).infer_objects(copy=False)
        # step 6.2: additional conversions based on the initialisation:
        self.helper_missing_values()
        self.df = (
            self.df.infer_objects()
        )  # try to infer what data types they actually are

        # step 7: get data types to save in self.dict_datatypes
        self.helper_feature_types()
        print(
            f"Data types of the features:\n"
            f"   categorical: {list(self.dict_datatypes.values()).count('cat')}\n"
            f"   datetime: {list(self.dict_datatypes.values()).count('datetime')}\n"
            f"   numerical: {list(self.dict_datatypes.values()).count('num')}"
        )

        # step 8: load thresholds for assessments
        print("Thresholds for assessment:")
        with open(self.path_thresholds) as file:
            list_thresholds = yaml.load(file, Loader=yaml.FullLoader)
        self.df_thresholds = pd.DataFrame(list_thresholds).T
        # format and print the thresholds
        print(
            self.df_thresholds.__str__()
            .replace("\n", "\n   ")
            .replace("good", "   good")
        )
        # process the thresholds
        for col in self.df_thresholds:
            self.df_thresholds[f"{col}_lower"] = self.df_thresholds[col].str.replace(
                "= x ", ""
            )
            self.df_thresholds[f"{col}_lower"] = self.df_thresholds[
                f"{col}_lower"
            ].str.split("<", expand=True)[0]
            self.df_thresholds[f"{col}_lower"] = (
                self.df_thresholds[f"{col}_lower"]
                .replace(["x", "x "], np.nan)
                .astype(float)
            )
        # create dict
        for x in self.df_thresholds.index:
            self.dict_thresholds[x] = [
                self.df_thresholds.loc[x, "good_lower"],
                self.df_thresholds.loc[x, "medium_lower"],
                self.df_thresholds.loc[x, "bad_lower"],
            ]

        # step 9: Final checks
        assert isinstance(self.feature_cols, list), "feature_cols is not a list"
        assert isinstance(self.key_cols, list), "key_cols is not a list"
        assert isinstance(self.excluded_cols, list), "excluded_cols is not a list"
        assert isinstance(
            self.excluded_feature_cols, list
        ), "excluded_feature_cols is not a list"
        assert (
            isinstance(self.date_col, str) or self.date_col is None
        ), "date_col is not a string or None"

    # ----------------------------------------------------------------------------- #
    # Helper functions
    # ----------------------------------------------------------------------------- #
    def helper_feature_types(self):
        """
        Is the risk driver categorical, datetime or numerical?
        """
        if self.force_datatypes is None:
            self.force_datatypes = []
            return
        elif isinstance(self.force_datatypes, list) and len(self.force_datatypes) == 0:
            return
        # if there are cases to check, loop over each column
        for col in self.feature_cols:
            try:
                self.dict_datatypes[col] = self.force_datatypes[col]
            except KeyError:
                if (
                    self.df.dtypes[col].name in self.datatypes_numeric
                ):  # the ones to be considered numeric
                    self.dict_datatypes[col] = "num"
                elif (
                    self.df.dtypes[col].name in self.datatypes_datetime
                ):  # the ones to be considered datetime
                    self.dict_datatypes[col] = "datetime"
                elif (
                    self.df.dtypes[col].name in self.datatypes_categorical
                ):  # the ones to be considered cat.
                    self.dict_datatypes[col] = "cat"
                else:
                    raise LookupError(
                        "Invalid datatype encountered, please adjust and run again"
                    )

    def helper_missing_values(self):
        """
        Impute interpretation of missing values. Additional values to be
        considered as NaNs can be forced by setting force_missing_values
        when initialising the class.
        """
        if self.force_missing_values is None:
            self.force_missing_values = []
            return
        elif isinstance(self.force_missing_values, list) and len(self.force_missing_values) == 0:
            return
        # if there are cases to check, loop over each column
        for col in self.df.columns:
            try:
                self.df[col] = (
                    self.df[col]
                    .replace(to_replace=self.force_missing_values[col], value=np.nan)
                    .infer_objects(copy=False)
                )
            except KeyError:
                pass

    def helper_resample_date_col(self) -> pd.DataFrame():
        """
        A helper function that resamples the date column (if it exists)
        and returns a data frame.
        """
        # copy data frame
        df0 = self.df.copy()

        # resample only if there is a date column and the frequency is not None
        if (self.date_col is not None) and (self.sample_freq is not None):
            # resample
            if self.sample_freq == "Y":
                df0[self.date_col] = df0[self.date_col].apply(lambda x: str(x)[:4])
            elif self.sample_freq == "M":
                df0[self.date_col] = df0[self.date_col].apply(lambda x: str(x)[:7])
            elif self.sample_freq == "D":
                df0[self.date_col] = df0[self.date_col].apply(lambda x: str(x)[:10])
            elif self.sample_freq == "H":
                df0[self.date_col] = df0[self.date_col].apply(lambda x: str(x)[:13])
            else:
                raise NotImplemented("Sampling frequency not valid.")
        else:
            pass

        return df0

    # ----------------------------------------------------------------------------- #
    # Tests
    # ----------------------------------------------------------------------------- #
    # 1: completeness / availability
    def analyse_completeness(
        self, analyse_total: bool = False, include_datetime_and_key_cols: bool = False
    ) -> None:
        """
        This method analyses the completeness / availability (missing values etc.)
        overall: If date_col/frequency is None or analyse_total is True
        over time: If self.date_col is not None or analyse_total is False

        Datetime column and key columns can be included in the analysis
        by setting include_datetime_and_key_cols=True.

        """
        if (
            self.date_col is None or self.sample_freq is None or analyse_total
        ):  # calculating overall completeness
            if include_datetime_and_key_cols:
                df = self.df.copy()
            else:
                df = self.df[self.feature_cols].copy()
            self.df_missing = pd.DataFrame(
                {"total missing": df.isna().sum() / len(df)}
            ).T
            self.df_completeness = 1 - self.df_missing.rename(
                {"total missing": "total available"}
            )
        else:
            # resample
            df0 = self.helper_resample_date_col()

            # group by datetime and calculate mean missing per group
            df0 = df0.set_index(self.date_col)
            df_missing = df0.isna()
            # save dataframes
            self.df_missing = df_missing.reset_index().groupby(self.date_col).mean()
            self.df_completeness = 1 - self.df_missing

    # 2: Timeliness
    def analyse_timeliness(self):
        """
        Analyses timeliness
        """
        # resample
        df0 = self.helper_resample_date_col()

    # 3: Accuracy
    def analyse_accuracy(self):
        pass

    # ----------------------------------------------------------------------------- #
    # Plots
    # ----------------------------------------------------------------------------- #
    # 1: Completeness
    def plot_completeness(
        self,
        analyse_total: bool = False,
        include_datetime_and_key_cols: bool = False,
        show: bool = False,
    ):
        """
        Plot completeness, give a dictionary of plots

        Future ideas:
            perhaps an axis/figure for each type
        """
        # 1: call calculation
        self.analyse_completeness(
            analyse_total=analyse_total,
            include_datetime_and_key_cols=include_datetime_and_key_cols,
        )
        if include_datetime_and_key_cols:
            str_title = "Total completeness for all features, datetime and key columns"
        else:
            str_title = "Total completeness for all features"
        if analyse_total:
            df0 = self.df_completeness.T
            fig = df0.plot(
                kind="bar",
                title=str_title,
                legend=False,
                ylim=[0, 1.0],
                figsize=(12, 8),
            )
            self.plots_completeness["total"] = fig
            if show:
                plt.show()

        else:
            for col in self.feature_cols:
                fig = self.df_completeness[col].plot(
                    title=f"{col}: Completeness rate",
                    legend=False,
                    ylim=[0, 1.0],
                    figsize=(12, 8),
                )
                self.plots_completeness[col] = fig
                if show:
                    plt.show()
