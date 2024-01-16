import pandas as pd
import numpy as np
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from scipy import stats
from itertools import combinations
from textwrap import fill
from tqdm import tqdm
from typing import Union
import joblib
import cuml

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

# Distribution
from scipy.stats import randint, uniform

# Preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import BaseCrossValidator
from cuml.preprocessing import SimpleImputer, StandardScaler, RobustScaler
from cuml.pipeline import Pipeline

# PCA
from cuml.decomposition import PCA

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Cluster (Outlier Detection)
from cuml.cluster import DBSCAN
from cuml.neighbors import NearestNeighbors

# Liner Regression
from cuml.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# Decision Tree
from cuml.ensemble import RandomForestRegressor  # Random Forest Baseline
# Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Metrics
from cuml.metrics import mean_squared_error, r2_score, mean_absolute_error

# Shap
import shap

# Plot Colors
PRIMARY_COLOR = '#1799E7'
SECONDARY_COLOR = 'black'
DASHED_LINE_COLOR = 'gray'


class Utils:
    features_similar = [['NetIncome', 'ROA', 'BasicEPS', 'ROE', 'Net Profit Margin'], ['Debt Ratio', 'Debt to Equity', 'Assets to Equity'], ['TotalRevenue', 'OperatingRevenue'], ['EBIT', 'Interest Coverage'], [
        'Current Ratio', 'Quick Ratio'], ['Cash Flow Margin', 'CROA', 'Cash Flow to Debt'], ['Tangible Book Value Per Share', 'Book Value Per Share'], ['gold', 'silver', 'platinum'], ['copper', 'aluminium'], ['wheat', 'corn', 'soybean']]

    features_econ = ['fed_funds', 'treasury_3mo', 'vix', 'gold', 'silver', 'platinum', 'oil', 'copper', 'aluminium', 'palladium', 'natural_gas', 'wheat', 'soybean', 'coffee', 'corn', 'sugar', 'cotton',
                     'unemployment', 'federal_budget_deficit', 'housing_starts', 'gdp_real', 'avg_home_price', '30yr-15yr Mortgage Rate Spread', 'BAA-AAA Corporate Bond Spread', '10yr-2yr Treasury Spread', 'cpi']

    def __init__(self):
        self.plot = Plot(self)
        self.stats = Stats(self)
        self.outlier = Outlier(self)

        # Display everything
        pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', )

        # Seeds
        np.random.seed(42)
        cp.random.seed(42)

    def init_df(self, df: pd.DataFrame):
        """
        Initializes the DataFrame with proper data types and returns it along with the features and target.
        """
        # Convert object cols and sector/industry IDs to category
        cat_cols = ['symbol', 'cap']
        df.drop(
            columns=[col for col in df.columns if 'ID' in col], inplace=True)
        cat_cols += [col for col in df.columns if 'GICS' in col]

        for col in cat_cols:
            df[col] = df[col].astype('category')

        # Convert date to datetime
        df['date'] = df['date'].astype('datetime64[s]')
        # Convert floats to 32bit for memory efficiency
        for col in df.select_dtypes(include='float64').columns:
            df[col] = df[col].astype('float32')

        self.target = df.select_dtypes(include='float32').columns[0]
        features, _ = self.features(df)
        return df, features, self.target

    def df_sector_info(self, df: pd.DataFrame):
        """
        Prints some useful sector and industry information from the DataFrame.
        """
        print('Shape:', df.shape)
        # Total number of earnings reports
        earnings = df['EBIT'].notna().sum()
        # Number of earnings
        print('Earnings:', earnings, f'({earnings / df.shape[0]:.2%})')

        # Number of unique symbols (companies)
        print('Companies:', df['symbol'].nunique())

        # Display by cap
        cap = df['cap'].value_counts().rename('Total').to_frame()
        cap['Total %'] = (cap['Total'] / df.shape[0] * 100).round(2)
        cap['Earnings'] = df.groupby(['cap'], observed=True)[
            'EBIT'].apply(lambda x: x.notna().sum())
        cap['Earnings %'] = (cap['Earnings'] / cap['Total'] * 100).round(2)
        cap['Earnings % of Total'] = (
            cap['Earnings'] / earnings * 100).round(2)
        display(cap)

        # Display GICS Industry Groups, Industries, and Sub-Industries
        cat = ['GICS Industry Group', 'GICS Industry', 'GICS Sub-Industry']
        for i, group in enumerate(cat):
            if i == 0:
                grouped = df.groupby([group], observed=True).size()
                pass
            else:
                grouped = df.groupby([cat[i-1], group], observed=True).size()
            grouped = grouped[grouped.ne(0)]
            grouped = grouped.to_frame().rename(columns={0: 'Total'})
            grouped['Total %'] = (
                grouped['Total'] / df.shape[0] * 100).round(2)
            earn = df.groupby([group], observed=True)['EBIT'].apply(
                lambda x: x.notna().sum()).to_frame().rename(columns={'EBIT': 'Earnings'})
            grouped = pd.merge(
                grouped, earn, left_index=True, right_index=True)
            grouped['Earnings %'] = (
                grouped['Earnings'] / grouped['Total'] * 100).round(2)
            grouped['Earnings % of Total'] = (
                grouped['Earnings'] / earnings * 100).round(2)
            display(grouped)

    def features(self, df: pd.DataFrame, lag: str = '_lag'):
        df = df.copy()
        """Returns the features and lag features (if any) from the DataFrame"""
        # Select float32 columns that are not the target
        features = np.array(df.drop(columns=[self.target]).select_dtypes(
            include='float32').columns)
        lag_features = np.array([] +
                                [feature for feature in features if lag in feature])
        features = np.array([
            feature for feature in features if feature not in lag_features])
        return features, lag_features

    def feat_nulls(self, df: pd.DataFrame):
        query = df.query('`EBIT`.notnull()')
        print(query.shape)
        nulls = query.isnull().sum().sort_values(
            ascending=False).rename('Nulls').to_frame()
        nulls['Null %'] = (nulls['Nulls'] / query.shape[0] * 100).round(2)
        display(nulls)

    @classmethod
    def lag_features(cls, df: pd.DataFrame, features, lag: int) -> pd.DataFrame:
        """
        Creates a lag feature for the specified features in the DataFrame.
        """
        df = df.sort_values(['symbol', 'date'])

        group = df.groupby('symbol', observed=True)
        gs = []
        for _, g in group:
            # Must resample to fill missing dates
            # this way lags will be accurate considering
            # that some companies have missing months
            g = g.set_index('date').resample('BME').asfreq()
            for feat in features:
                g[f'{feat}_lag{lag}'] = g[feat].shift(lag)
            # Drop null symbols
            g = g[g['symbol'].notna()]
            gs.append(g.reset_index())
        return pd.concat(gs, ignore_index=True)


class Stats(Utils):
    def __init__(self, utils: Utils):
        self.utils = utils

    @classmethod
    def vif(cls, df: pd.DataFrame):
        X = df.copy()
        """Runs VIF on the dataframe to check for multicollinearity, returning resulting VIF values."""
        # Impute
        from sklearn.impute import SimpleImputer
        imputed = SimpleImputer(strategy='constant',
                                fill_value=0).fit_transform(X)
        # Scale
        from sklearn.preprocessing import StandardScaler
        scaled = StandardScaler().fit_transform(imputed)

        vif_df = pd.DataFrame(scaled, columns=df.columns)
        # Add constant to features
        X: pd.DataFrame = add_constant(vif_df)
        # Calculate VIF
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i)
                      for i in range(X.shape[1])]
        vif = vif.sort_values(by='VIF', ascending=False)

        # Drop constant
        vif = vif[vif['Features'] != 'const']
        return vif

    @classmethod
    def winsorize(cls, df: Union[pd.DataFrame, pd.Series], iqr: float = 1.5):
        """
        Winsorizes the DataFrame or Series using the IQR method and returns the resulting DataFrame or Series.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
            is_series = True
        else:
            is_series = False

        df = df.copy()
        # Calculate the quantiles
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        # Calculate the IQR
        _iqr = q3 - q1
        # Calculate the lower and upper bounds
        lower_bound = q1 - _iqr * iqr
        upper_bound = q3 + _iqr * iqr
        # Apply the winsorization
        for col in df.columns:
            df[col] = df[col].clip(lower_bound[col], upper_bound[col])

        if is_series:
            return df.iloc[:, 0]
        else:
            return df


class Plot(Utils):
    def __init__(self, utils: Utils):
        self.utils = utils

    @classmethod
    def acf(cls, df, column, lags=36):
        """
        Displays the Autocorrelation Function (ACF) plot for the aggregated values of the specified column.

        For months with multiple reports, the median value is used and missing values are forward filled.
        """
        num_plots = 1
        axes = 0
        num_cols = 2  # Number of columns for the subplot grid
        if isinstance(column, list) or isinstance(column, np.ndarray):
            num_plots = len(column)
            # Calculate number of rows needed
            num_rows: int = (num_plots + num_cols - 1) // num_cols
            _, axes = plt.subplots(num_rows, num_cols, figsize=(
                8 * num_cols, 4 * num_rows), tight_layout=True)
            axes = axes.flatten()

        # Create ACF plot
        for i, col in enumerate(column if num_plots > 1 else [column]):
            # Aggregate data by date
            df_agg = df.groupby('date').agg({col: 'median'})
            # Forward fill missing values
            df_agg = df_agg.ffill()

            ax = axes[i] if num_plots > 1 else None
            sm.graphics.tsa.plot_acf(df_agg, lags=lags, ax=ax)
            if ax is None:
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.title('Autocorrelation Function (ACF) Plot {}'.format(col))
                plt.grid(True)
                plt.show()
                return
            else:
                ax.set_xlabel('Lag')
                ax.set_ylabel('Autocorrelation')
                ax.set_title(
                    'Autocorrelation Function (ACF) Plot {}'.format(col))
                ax.grid(True)

        plt.show()

    @classmethod
    def dist(cls, df: pd.DataFrame):
        """Displays the distribution of the target column."""
        # Create figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Plot distribution of target
        sns.kdeplot(df, color=PRIMARY_COLOR, fill=True, ax=axs[0])
        axs[0].set_title('KDE Distribution', pad=20, fontweight='bold')

        # Plot boxplot of target
        sns.boxplot(x=df, color=PRIMARY_COLOR, ax=axs[1])
        axs[1].set_title('Box Plot', pad=20, fontweight='bold')

        plt.tight_layout()
        plt.show()

    @classmethod
    def correlation_matrix(cls, df: pd.DataFrame, sort: bool = True, lower_bound: float = 0.6, display_scatter: bool = True):
        """Displays the correlation matrix and scatter regression plots of symmetrically and asymmetrically correlated features."""
        correlation_matrix = df.corr()

        # Plot correlation matrix
        # TODO make size dynamic
        plt.figure(figsize=(35, 32))
        sns.heatmap(correlation_matrix, annot=True,
                    cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Correlation Matrix', fontsize=30, pad=30, fontweight='bold')
        plt.show()

        if not display_scatter:
            return

        # Lists to store symmetrically and asymmetrically correlated features
        symmetric_correlations = []
        asymmetric_correlations = []
        # Iterate through each feature
        for feature1 in correlation_matrix.columns:
            # Iterate through remaining features
            for feature2 in correlation_matrix.columns:
                # Skip self-correlation
                if feature1 == feature2:
                    continue
                # Get correlation between the two features
                correlation = correlation_matrix.loc[feature1, feature2]
                # Check if correlation is greater than the cut-off
                if abs(correlation) >= lower_bound:
                    if ((feature2, feature1, correlation) not in symmetric_correlations) and \
                            ('_lag' not in feature2 and feature1):
                        # Check if correlation is symmetric
                        if correlation_matrix.loc[feature2, feature1] == correlation:
                            symmetric_correlations.append(
                                (feature1, feature2, correlation))
                        else:
                            asymmetric_correlations.append(
                                (feature1, feature2, correlation))

        for correlations in zip([symmetric_correlations, asymmetric_correlations],
                                ['Symmetrically Correlated Features:', 'Asymmetrically Correlated Features:']):

            if correlations[0]:
                if sort:
                    correlations[0].sort(key=lambda x: abs(x[2]), reverse=True)
                num_plots = len(correlations[0])
                num_cols = 3  # Number of columns for subplots
                num_rows = (num_plots + num_cols - 1) // num_cols
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(
                    15, 5 * num_rows), layout='constrained')
                fig.suptitle(
                    f'Scatter Plots of {correlations[1]} (> {lower_bound})', fontsize=20, fontweight='bold')
                axes = axes.flatten()

                for i, (feature1, feature2, correlation) in enumerate(correlations[0]):
                    ax = axes[i]
                    df_1 = df[[feature1, feature2]].dropna()
                    X, y = df_1[feature1], df_1[feature2]
                    model = LinearRegression(copy_X=True, algorithm='svd')
                    model.fit(X, y)

                    # Create range for regression line
                    x_range = np.linspace(X.min(), X.max(), 100)
                    y_range = model.predict(x_range)

                    # Calculate standard errors
                    y_pred = model.predict(X)
                    residuals = y - y_pred
                    residual_sum_of_squares = residuals.T @ residuals
                    sigma_squared_hat = residual_sum_of_squares / \
                        (X.shape[0] - 1)
                    X_with_intercept = np.c_[np.ones(X.shape[0]), X]
                    var_beta_hat = sigma_squared_hat * \
                        np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                    se_beta_hat = np.sqrt(np.diag(var_beta_hat))

                    # Calculate confidence intervals
                    t_value = stats.t.ppf(1 - 0.05 / 2, df=X.shape[0] - 1)
                    # Use se_beta_hat[1] because it corresponds to the slope of the regression line
                    confidence_interval = t_value * se_beta_hat[1]

                    # Plot scatter plot, regression line, and confidence intervals
                    ax.scatter(X, y, color=PRIMARY_COLOR, alpha=0.7)
                    ax.plot(x_range, y_range, color=SECONDARY_COLOR, zorder=999)
                    ax.fill_between(x_range, (y_range - confidence_interval),
                                    (y_range + confidence_interval), color=SECONDARY_COLOR, alpha=0.2)
                    ax.set_title(fill(
                        f'{feature1} vs {feature2} (Correlation: {correlation:.2f})', 50), wrap=True)

                # Extra spacing between subplots
                fig.get_layout_engine().set(hspace=0.1, wspace=0.1)
                plt.show()
                print('\n')
            else:
                print(
                    f'{correlations[1]} None with correlation >= {lower_bound}.')

    @classmethod
    def yearly_lag(cls, df: pd.DataFrame, feature: str, n_lags: int = 1):
        # Agg by date
        df_agg = df.groupby('date').agg({feature: 'median'})
        # Forward fill
        df_agg = df_agg.ffill()
        # Create lag features
        for i in range(1, n_lags + 1):
            df_agg[f'{feature}_lag_{i}'] = df_agg[feature].shift(i * 12)
        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df_agg.index, df_agg[feature], label=feature)
        for i in range(1, n_lags + 1):
            ax.plot(df_agg.index,
                    df_agg[f'{feature}_lag_{i}'], label=f'{feature} Lag {i}')
        ax.set_title(f'{feature} and {feature} Lags')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.show()


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        elif self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]


class MulticollinearityCV:
    """
    Addresses multicollinearity by seeking the best subset of high VIF 
    features using cross-validation, maximizing the mean R2 score.
    """

    def __init__(self, estimator, cv, keep_max=1, problems=None):
        self.estimator = estimator
        self.cv = cv
        self.keep_max = keep_max
        self.problems = Utils.features_similar.copy(
        ) if problems is None else problems.copy()

    def fit(self, X, y):
        x_ = X.copy()
        # Account for lag similar features
        sim_feats = self.problems
        sim_feats_lagged = [[f'{feature}_lag{i}' for feature in group]
                            for group in sim_feats for i in range(1, 13)]
        sim_feats += sim_feats_lagged
        # Only keep features that are in the DataFrame
        sim_feats = [[feature for feature in group if feature in x_.columns]
                     for group in sim_feats]
        # Remove empty groups
        sim_feats = [group for group in sim_feats if group]
        print(f'Checking Features: {sim_feats}')
        for group in sim_feats:
            # Exhastive search for the best subset of features
            baseline = np.mean(cross_val_score(
                self.estimator, x_, y, cv=self.cv, scoring='r2'))
            print(f'Baseline: {baseline:.4%}')
            subset_scores = {}
            for combo in tqdm([combo for r in range(0, self.keep_max + 1) for combo in combinations(group, r)]):
                x_subset = x_[
                    [feature for feature in x_.columns if feature not in group] + list(combo)]
                score_ = np.mean(cross_val_score(
                    self.estimator, x_subset, y, cv=self.cv, scoring='r2'))
                subset_scores[combo] = score_
            # Sort the scores
            subset_scores = {k: v for k, v in sorted(
                subset_scores.items(), key=lambda item: item[1], reverse=True)}
            # Get the subset and score that has less than keep_max features
            subset_scores = {
                k: v for k, v in subset_scores.items() if len(k) <= self.keep_max}
            # Get the score of the highst
            best_score = list(subset_scores.values())[0]
            print('Keep:', list(subset_scores.keys())[0])
            if best_score < baseline:
                print('Best Subset Loss:',  f'{baseline - best_score:.4%}')
            else:
                print('Best Subset Gain:',  f'{best_score - baseline:.4%}')
            # Drop features that are not in the best subset
            best_subset = list(subset_scores.keys())[0]
            x_ = x_[
                [feature for feature in x_.columns if feature not in group] + list(best_subset)]
        self.selected_features_ = x_.columns


def cross_val_score(model, x: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator, scoring='r2'):
    if scoring == 'r2':
        scoring = r2_score
    elif scoring == 'mae':
        scoring = mean_absolute_error
    elif scoring == 'mse':
        scoring = mean_squared_error

    scores = []
    for train_index, test_index in cv.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        try:
            with cuml.using_output_type('numpy'):
                model.fit(cp.asarray(x_train.values),
                          cp.asarray(y_train.values))
                scores.append(scoring(cp.asarry(y_test.values),
                              model.predict(cp.asarray(x_test.values))))
        except:
            model.fit(x_train.values, y_train.values)
            scores.append(scoring(y_test.values, model.predict(x_test.values)))
    return scores if isinstance(scores, np.ndarray) else cp.asnumpy(cp.array(scores))


class CustomCV(BaseCrossValidator):
    def __init__(self, mask: pd.Series, num_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.num_splits = num_splits
        self.shuffle = shuffle
        self.random_state = random_state
        # Mask for earnings dates (1 if missing, 0 if not)
        self.mask = mask.isna().astype(int)

    def split(self, X, y=None, groups=None):
        """
        Generates indices to split data into training and test set.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        skv = StratifiedKFold(
            n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.random_state)

        for train_index, test_index in skv.split(X, self.mask):
            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator
        """
        return self.num_splits


class Tuner:
    """
    RandomizedSearchCV wrapper class for hyperparameter tuning using cross-validation.
    """
    gbt_param_dist = {
        # Constants
        'device': ['gpu'],                           # Device to use
        # Seed for the random number generator
        'random_state': [42],
        'verbose': [-1],                              # Verbosity of the model
        # Tunable
        # Number of leaves in one tree
        'num_leaves': randint(6, 50),
        'max_depth': randint(3, 20),                 # Maximum depth of a tree
        'learning_rate': uniform(loc=0.001, scale=0.5),  # Learning rate
        # Number of boosting iterations
        'n_estimators': randint(50, 1000),
        # Subsample ratio of the training instance
        'subsample': uniform(loc=0.5, scale=0.5),
        # Subsample ratio of columns when constructing each tree
        'colsample_bytree': uniform(loc=0.5, scale=0.5),
        # L1 regularization term on weights
        'reg_alpha': uniform(loc=0, scale=1),
        # L2 regularization term on weights
        'reg_lambda': uniform(loc=0, scale=1),
        # Minimum number of data needed in a child (leaf)
        'min_child_samples': randint(20, 100),
        # Minimum sum of instance weight (hessian) needed in a child
        'min_child_weight': uniform(loc=0.001, scale=0.1),
    }

    elasticnet_param_dist = {
        # Constants
        # Whether to calculate the intercept for this model
        'fit_intercept': [True],
        # Already normalizing in the pipeline
        'normalize': [False],
        # Tunable
        # Constant that multiplies the penalty terms (0 is equivalent to unpenalized model (OLS))
        'alpha': uniform(loc=0, scale=1),
        # The ElasticNet mixing parameter (0 is L2, 1 is L1)
        'l1_ratio': uniform(loc=0, scale=1),
    }

    def __init__(self, estimator, preprocessor, param_dist, cv, n_iter=100, random_state=42, verbose=1, scoring='r2'):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.param_dist = param_dist
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.scoring = scoring

    def fit(self, X, y):
        random_search = RandomizedSearchCV(estimator=self.estimator, param_distributions=self.param_dist,
                                           n_iter=self.n_iter, cv=self.cv, random_state=self.random_state, verbose=self.verbose, scoring=self.scoring)
        X_prep = X.copy()
        X_prep = self.preprocessor.fit_transform(X.values)
        random_search.fit(X_prep, y)
        self.best_params_ = random_search.best_params_
        self.best_estimator_ = random_search.best_estimator_
        del random_search
        return self

    def save_params(self, path):
        joblib.dump(self.best_params_, path)
        return self

    @classmethod
    def load_params(cls, path):
        return joblib.load(path)


class RFECV:
    """
    Recursive Feature Elimination (RFE) class for feature
    selection using cross-validation.

    This RFE iteratively removes features until the mean score is maximized, not just
    dropping the least important feature each time.
    """
    support_ = None

    def __init__(self, estimator, cv: BaseCrossValidator = None, scoring='r2'):
        self.estimator = estimator
        self.cv = cv
        self._scoring = scoring

        if scoring == 'r2':
            self._metric = r2_score
            self._best = max
        elif scoring == 'mae':
            self._metric = mean_absolute_error
            self._best = min
        elif scoring == 'mse':
            self._metric = mean_squared_error
            self._best = min

        # Set the metric and its comparison function
        # for r2 you want a higher score while for mae and mse you want a lower score
        self._compare = lambda x, y: x >= y
        if scoring != 'r2':
            self._compare = lambda x, y: x <= y

    def fit(self, x, y):
        """Fits the RFE model to the data."""
        self.support_ = pd.DataFrame(
            True, index=x.columns, columns=['Support'])
        x = x.copy()
        pipeline = self.estimator
        best_score = np.mean(cross_val_score(
            pipeline, x, y, cv=self.cv, scoring=self._scoring))
        print(f'Initial r2 score: {best_score:.4%}')
        print(
            f'Initial mae score: {np.mean(cross_val_score(pipeline, x, y, cv=self.cv, scoring="mae")):.4%}')
        # 1. Test dropping each feature 1 by 1
        while True:
            scores = []
            cols = np.array(x.columns.to_list())
            np.random.shuffle(cols)
            for feature in cols:
                x_test = x.drop(columns=[feature])
                if self.cv:
                    score = cross_val_score(
                        pipeline, x_test, y, cv=self.cv, scoring=self._scoring)
                    scores.append((np.mean(score), feature))
                else:
                    pipeline.fit(cp.asarray(x_test.values),
                                 cp.asarray(y.values))
                    scores.append(
                        (self._metric(y.values, pipeline.predict(cp.asarray(x_test.values))), feature))
            # Get the best score and its corresponding dropped feature
            best = self._best(scores, key=lambda x: x[0])
            if self._compare(best[0], best_score):
                best_score = best[0]
                # Drop the feature
                x.drop(columns=[best[1]], inplace=True)
                self.support_.loc[best[1], 'Support'] = False
                # print(f'Score: {best_score:.4%} With Dropped: {best[1]}...')
            else:
                print(f'New r2 score: {best_score:.4%}')
                print(
                    f'New mae score: {np.mean(cross_val_score(pipeline, x, y, cv=self.cv, scoring="mae")):.4%}')
                break
            del scores, best
        del best_score
        return self


class Outlier(Utils):
    def __init__(self, utils: Utils):
        self.utils = utils

    def knee_plot(self, x, n_neighbors: int = 10):
        """
        Displays the Knee Plot for DBSCAN's epsilon hyperparameter using the k-distance graph.
        """
        x = x.copy()
        # Impute and scale
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        scaler = StandardScaler()

        x = imputer.fit_transform(x.values)
        x = scaler.fit_transform(x)

        # Fit the nearest neighbors estimator to the data
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(x)  # Removed reshape here

        # Calculate the distance to the kth nearest neighbor for each point
        distances, _ = nn.kneighbors(x)

        # Sort the distances
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # Create an interactive plot
        fig = px.line(x=range(len(distances)), y=distances, labels={'x': 'Points sorted according to distance of '+str(
            n_neighbors)+'th nearest neighbor', 'y': str(n_neighbors)+'th nearest neighbor distance'}, title='K-distance Graph')
        fig.show()

    def dbscan(self, df, features, eps: float, min_samples: int = 4):
        """
        Applies DBSCAN on the specified DataFrame and returns the labels.
        """
        x = df.copy()
        # Impute and scale
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        scaler = StandardScaler()

        x = imputer.fit_transform(df[features].values)
        x = scaler.fit_transform(x)

        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(x)
        df['labels'] = labels

        return df


def coef_significance(model, X, y, y_pred):
    """
    Returns coef values, standard error, t-stat, and p-values of the coefficients of the specified model.
    """
    if not isinstance(y, cp.ndarray):
        y = cp.asarray(y)
    if not isinstance(y_pred, cp.ndarray):
        y_pred = cp.asarray(y_pred)
    X = X.copy()
    X_transformed = model[:-1].fit_transform(X.values)
    X_transformed = cp.asarray(X_transformed)
    deg_fr = X_transformed.shape[0] - X_transformed.shape[1] - 1

    # Grab the coefficients
    coef = cp.append(model[-1].coef_, model[-1].intercept_)
    # Add a column of ones for the intercept
    X_transformed = cp.hstack(
        (cp.ones((X_transformed.shape[0], 1)), X_transformed))

    # Calculate the mean squared error
    mse = cp.sum((y - y_pred)**2) / deg_fr
    # Calculate the variance of the coefficients
    variance_coef = cp.linalg.inv(
        cp.dot(X_transformed.T, X_transformed)).diagonal()
    # Calculate the standard error
    std_err = cp.sqrt(variance_coef * mse)
    # Calculate the t-statistic
    t_stat = (coef / std_err).get()

    # Calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), deg_fr))

    del X, X_transformed, deg_fr, mse, variance_coef
    return coef.get(), std_err.get(), t_stat, p_value


class Evaluate:
    def __init__(self, estimator, cv, decimals=2):
        self.estimator = estimator
        self.cv = cv
        self.decimals = decimals
        self._metrics_cv_ = None
        self._metrics_ = None
        self._median_model = None

    def fit(self, X, y):
        self._metrics_cv_ = self._calc_metrics_cv(X.copy(), y)
        self._metrics_ = self._calc_metrics(X.copy(), y)

    def _calc_metrics_cv(self, X, y):
        metrics_cv = pd.DataFrame(
            columns=['Median', 'Mean', 'STD', 'Min', 'Max'])
        scores = cp.empty((7, self.cv.get_n_splits()), dtype=cp.float32)
        models = {}
        for idx, (train, test) in enumerate(self.cv.split(X, y)):
            x_train, x_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            try:
                self.estimator.fit(cp.asarray(x_train.values),
                                   cp.asarray(y_train.values))
                y_pred = self.estimator.predict(
                    cp.asarray(cp.asarray(x_test.values)))
            except:
                self.estimator.fit(x_train.values, y_train.values)
                y_pred = self.estimator.predict(x_test.values)
            n, p = x_train.shape
            # Degrees of freedom
            # The linear regression model has p + 1 parameters (p features + 1 intercept)
            if hasattr(self.estimator[-1], 'coef_'):
                deg_fr = n - p - 1
            else:
                deg_fr = n - p
            y_test = cp.asarray(y_test)
            y_pred = cp.asarray(y_pred)
            scores[0, idx] = r2_score(y_test, y_pred)
            # Adjusted R2
            scores[1, idx] = 1 - (1 - scores[0, idx]) * (n - 1) / deg_fr
            scores[2, idx] = mean_squared_error(y_test, y_pred)
            scores[3, idx] = cp.sqrt(scores[2, idx])
            scores[4, idx] = mean_absolute_error(y_test, y_pred)
            # SSR and SSE
            ssr = cp.sum((y_pred - y_test.mean())**2)
            sse = cp.sum((y_test - y_pred)**2)
            # F-statistic
            scores[5, idx] = (ssr / p) / (sse / deg_fr)
            # (F-stat) P-value
            scores[6, idx] = 1 - stats.f.cdf(scores[5, idx].get(), p, deg_fr)

            # Save the current fitted model
            models[float(scores[0, idx])] = (self.estimator,
                                             x_train.copy(), y_test.copy(), y_pred.copy())
        # Calculate the median, mean, std, min, and max
        metrics_cv.loc['R2'] = [cp.median(scores[0]).get(), cp.mean(scores[0]).get(
        ), cp.std(scores[0]).get(), cp.min(scores[0]).get(), cp.max(scores[0]).get()]
        metrics_cv.loc['Adjusted R2'] = [cp.median(scores[1]).get(), cp.mean(scores[1]).get(
        ), cp.std(scores[1]).get(), cp.min(scores[1]).get(), cp.max(scores[1]).get()]
        metrics_cv.loc['MSE'] = [cp.median(scores[2]).get(), cp.mean(scores[2]).get(
        ), cp.std(scores[2]).get(), cp.min(scores[2]).get(), cp.max(scores[2]).get()]
        metrics_cv.loc['RMSE'] = [cp.median(scores[3]).get(), cp.mean(scores[3]).get(
        ), cp.std(scores[3]).get(), cp.min(scores[3]).get(), cp.max(scores[3]).get()]
        metrics_cv.loc['MAE'] = [cp.median(scores[4]).get(), cp.mean(scores[4]).get(
        ), cp.std(scores[4]).get(), cp.min(scores[4]).get(), cp.max(scores[4]).get()]
        metrics_cv.loc['F-stat'] = [cp.median(scores[5]).get(), cp.mean(scores[5]).get(
        ), cp.std(scores[5]).get(), cp.min(scores[5]).get(), cp.max(scores[5]).get()]
        metrics_cv.loc['P>F'] = [cp.median(scores[6]).get(), cp.mean(scores[6]).get(
        ), cp.std(scores[6]).get(), cp.min(scores[6]).get(), cp.max(scores[6]).get()]
        # Set the median model
        med_model = models[float(metrics_cv.loc['R2']['Median'])]
        self._median_model = med_model[0]
        self._X = med_model[1]
        self._y_true = cp.asarray(med_model[2])
        self._y_pred = cp.asarray(med_model[3])
        del scores, models, med_model
        return metrics_cv

    def _calc_metrics(self, X, y):
        """
        Calculate the metrics.
        """
        try:
            self.estimator.fit(cp.asarray(X.values), cp.asarray(y.values))
            y_pred = self.estimator.predict(cp.asarray(X.values))
        except:
            self.estimator.fit(X.values, y.values)
            y_pred = self.estimator.predict(X.values)
        n, p = X.shape
        # Degrees of freedom
        # The linear regression model has p + 1 parameters (p features + 1 intercept)
        if hasattr(self.estimator[-1], 'coef_'):
            deg_fr = n - p - 1
        else:
            deg_fr = n - p
        y_pred = cp.asarray(y_pred)
        y = cp.asarray(y)
        metrics = pd.DataFrame(columns=['Score'])
        metrics.loc['R2'] = r2_score(y, y_pred)
        metrics.loc['Adjusted R2'] = 1 - \
            (1 - metrics.loc['R2']) * (n - 1) / deg_fr
        metrics.loc['MSE'] = mean_squared_error(y, y_pred)
        metrics.loc['RMSE'] = metrics.loc['MSE'].map(np.sqrt)
        metrics.loc['MAE'] = mean_absolute_error(y, y_pred)
        ssr = cp.sum((y_pred - y.mean())**2)
        sse = cp.sum((y - y_pred)**2)
        metrics.loc['F-stat'] = (ssr / p) / (sse / deg_fr)
        metrics.loc['P>F'] = 1 - \
            stats.f.cdf(metrics.loc['F-stat']['Score'].get(), p, deg_fr)
        return metrics

    def linear_stats(self):
        X = self._X.copy()
        if self._metrics_ is None:
            raise ValueError('Please fit the model first.')
        if not hasattr(self.estimator[-1], 'coef_'):
            raise ValueError('The estimator does not have a coef_ attribute.')
        model = self._median_model

        alphas = [0.1, 0.05, 0.01]
        linear_stats = pd.DataFrame(index=X.columns.to_list() + ['Intercept'])

        coef, std_err, t_stat, p_value = coef_significance(
            model, X, self._y_true, self._y_pred)

        linear_stats['Coef'] = coef
        linear_stats['Std Err'] = std_err
        linear_stats['|t|'] = np.abs(t_stat)
        linear_stats[f'P>|t|'] = p_value
        for idx in range(linear_stats.shape[0]):
            if linear_stats.iloc[idx]['P>|t|'] < 0.01:
                star = '***'
            elif linear_stats.iloc[idx]['P>|t|'] < 0.05:
                star = '**'
            elif linear_stats.iloc[idx]['P>|t|'] < 0.1:
                star = '*'
            else:
                star = ''
            linear_stats.index.values[idx] = linear_stats.index.values[idx] + star

        for alpha in alphas:
            linear_stats[f't* ({alpha:.0%})'] = stats.t.ppf(1 -
                                                            alpha / 2, X.shape[0] - X.shape[1] - 1)
            # Assuming linear_stats is a DataFrame with 34 rows
            linear_stats[f'CI ({alpha:.0%})'] = linear_stats[f't* ({alpha:.0%})'] * \
                linear_stats['Std Err']
        # Drop the intercept
        linear_stats = linear_stats.drop(index=linear_stats.index[-1])
        # Sort the coefficients by their absolute value
        linear_stats = linear_stats.reindex(
            linear_stats['Coef'].abs().sort_values(ascending=False).index)
        del X
        return linear_stats

    def importance_plot(self, title=None):
        if self._metrics_ is None:
            raise ValueError(
                'The metrics have not been calculated, fit the model first.')
        if hasattr(self.estimator[-1], 'feature_importances_'):
            self._importance_plot(title)
        elif hasattr(self.estimator[-1], 'coef_'):
            self._coef_plot(title)
        else:
            raise ValueError(
                'The estimator does not have a feature_importances_ or coef_ attribute.')

    def _importance_plot(self, title=None):
        """
        Plot the feature importances.
        """
        if title is None:
            title = 'Feature Importance Plot'
        feature_importances = pd.Series(
            self._median_model[-1].feature_importances_, index=self._X.columns)
        feature_importances = feature_importances.sort_values()
        feature_importances.plot(
            kind='barh', color=PRIMARY_COLOR, figsize=(12, 8))
        plt.title(title, fontsize=20, fontweight='bold')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.grid(True)
        plt.show()

    def _coef_plot(self, title=None):
        """
        Plot the coefficients.
        """
        if title is None:
            title = 'Coefficients Plot'
        stats = self.linear_stats()[::-1]
        palette = ['#DC267F', '#FFB000', '#648FFF']
        plt.figure(figsize=(15, 10))
        for i, alpha in enumerate([0.1, 0.05, 0.01][::-1]):
            alpha_str = "{:.0%}".format(alpha)  # Format alpha as a percentage
            # stats[f'CI ({alpha_str})'] = pd.to_numeric(stats[f'CI ({alpha_str})'], errors='coerce')
            plt.errorbar(stats['Coef'], stats.index, xerr=stats[f'CI ({alpha_str})'], fmt='s',
                         label=f'CI {alpha_str}', color=palette[i])
        plt.axvline(0, color=SECONDARY_COLOR, linestyle='--', linewidth=2)
        plt.legend()
        plt.title(title, fontsize=20, fontweight='bold')
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.text(0.5, -0.1, 'Significance: *** <= 1%, ** <= 5%, * <= 10%',
                 ha='center', transform=plt.gca().transAxes)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('temp/linear-coef.png')
        plt.show()
        del stats

    @property
    def metrics_cv_(self):
        """
        Returns the metrics in a readable format as a pandas DataFrame.
        """
        if self._metrics_cv_ is None:
            raise ValueError(
                'The metrics have not been calculated, fit the model first.')
        metrics = self._metrics_cv_.copy()
        metrics[:5] = metrics[:5].map(
            lambda x: '{:.{}%}'.format(x, self.decimals))
        metrics[5:-1] = metrics[5:-1].map(
            lambda x: '{:.{}f}'.format(x, self.decimals))
        metrics[-1:] = metrics[-1:].map(
            lambda x: '{:.{}f}'.format(x, self.decimals + 2))
        return metrics

    @metrics_cv_.setter
    def metrics_cv_(self, value):
        self._metrics_cv_ = value

    @property
    def metrics_(self):
        """
        Returns the metrics in a readable format as a pandas DataFrame.
        """
        if self._metrics_ is None:
            raise ValueError(
                'The metrics have not been calculated, fit the model first.')
        metrics = self._metrics_.copy()
        metrics[:5] = metrics[:5].map(
            lambda x: '{:.{}%}'.format(x, self.decimals))
        metrics[5:-1] = metrics[5:-1].map(
            lambda x: '{:.{}f}'.format(x, self.decimals))
        metrics[-1:] = metrics[-1:].map(
            lambda x: '{:.{}f}'.format(x, self.decimals + 2))
        return metrics

    @metrics_.setter
    def metrics_(self, value):
        self._metrics_ = value

    def residuals_plot(self, title='Residuals Plot'):
        """
        Plot the residuals.
        """
        if self._metrics_ is None:
            raise ValueError(
                'The metrics have not been calculated, fit the model first.')
        y_pred = self._y_pred.get()
        y_true = self._y_true.get()
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
        fig.suptitle(title, fontsize=20, fontweight='bold')
        # Plot Results
        ax[0].scatter(y_true, y_pred, color=PRIMARY_COLOR, alpha=0.7)
        ax[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                   color=SECONDARY_COLOR, linestyle='--', linewidth=2)
        ax[0].set_title('Predicted vs True')
        ax[0].set_xlabel('True')
        ax[0].set_ylabel('Predicted')
        ax[0].grid(True)
        # QQ plot
        residuals = y_true - y_pred
        standardized_residuals = (
            residuals - residuals.mean()) / residuals.std()
        stats.probplot(standardized_residuals, dist='norm', plot=ax[1])
        ax[1].set_title(f'Normal Q-Q ($\sigma$ = {residuals.std():.4f})')
        ax[1].set_xlabel('Theoretical Quantiles ($\sigma$)')
        ax[1].set_ylabel('Standardized Observed Residuals ($\sigma$)')
        if 'LightGBM' in title:
            plt.savefig('temp/gbdt-resid.png')
        else:
            plt.savefig('temp/linear-resid.png')
        plt.show()

    def shap_plot(self):
        if self._metrics_ is None:
            raise ValueError(
                'The metrics have not been calculated, fit the model first.')
        if hasattr(self.estimator[-1], 'feature_importances_'):
            explainer = shap.GPUTreeExplainer
        elif hasattr(self.estimator[-1], 'coef_'):
            explainer = shap.LinearExplainer
        else:
            raise ValueError(
                'The estimator does not have a feature_importances_ or coef_ attribute.')
        X = self._X.copy()
        with cuml.using_output_type('numpy'):
            model = self._median_model
            data = model[:-1].fit_transform(X.values.copy())
            explainer = explainer(model[-1], data, random_state=42)
            data = pd.DataFrame(data, columns=X.columns.copy())
            shap_values_exp = explainer(data)
            shap_values = explainer.shap_values(data)
            fig = shap.plots.beeswarm(
                shap_values_exp, max_display=self._X.shape[1], show=False).figure
            fig.tight_layout()
            fig.savefig('temp/gbdt-shap.png')
            fig.show()
