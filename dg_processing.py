# visualization
import matplotlib.pyplot as plt
import matplotlib # for LaTeX display

# data processing & statistics
import numpy as np, pandas as pd
import scipy.stats

#
from sklearn import metrics, model_selection

# processing molecules
from rdkit import Chem

def log_transform(y, y0_target=0):
    '''
    Apply a log transformation `f` to the vector `y`, such that:
    - f(y0) = y0_target, where y0 = min(y)
    Inputs:
        y: Pandas Series, vector to apply transformation to
        y0_target: what to map the min element to
    '''
    y0 = min(y)
    return np.log10(y.values.astype(np.float) - y0 + 10**y0_target)

class LogTransformer:
    def __init__(self, cols_to_apply, y0_targets=None):
        '''
        Constructor for a `LogTransformer` which will apply a `log_transform`
        to the columns in `cols_to_apply`. A target for the minimal element in
        the columns can optionally be specified in `y0_targets`, which must have
        length the same as `cols_to_apply`.
        Inputs:
            cols_to_apply: list of string, specifying which columns to apply the
                log transforms to.
            y0_targets: optional list of reals, the target values for the minimal
                element in each column in `cols_to_apply`. If not `None`, then
                must have length the same as `cols_to_apply`.
        '''
        self.cols_to_apply = cols_to_apply

        if y0_targets:
            if len(cols_to_apply) != len(y0_targets):
                raise ValueError(f'y0_targets has length {len(y0_targets)}, which '
                                 f'is different to `cols_to_apply` with length'
                                 f'{len(cols_to_apply)}.')
            self.y0_targets = y0_targets
        else:
            self.y0_targets = [0.0]*len(cols_to_apply)

    def fit(self, X, y=None):
        '''
        Dummy class, for compatability with sklearn Pipelines objects
        '''
        return self

    def transform(self, df):
        '''
        Transforms `df`'s columns. Columns in `cols_to_apply` will be transformed.
        If `y0_targets` are defined, then the `i`th columns in `cols_to_apply` will
        have their min element mapped `y0_targets[i]`.
        Inputs:
            df: DataFrame, containing the data we'd like to transform. Must have all
                the columns in `cols_to_apply`
        '''
        out_df = df.copy()
        for col, target in zip(self.cols_to_apply, self.y0_targets):
            out_df[col] = log_transform(df[col], y0_target=target)
        return out_df

def feasibility_vector(losses, loss_cutoff, col_name=None):
    '''
    Defines the 'feasibility' of a molecule based on its loss.
    If the molecule failed DG conformation, then its loss will be NaN;
    else if it does, the `loss` will be its DG error function
    Input:
        losses: pandas series, of the `loss` column
    '''
    bad_mols = (losses.isnull() | (losses > loss_cutoff))
    loss_str = col_name if col_name else 'loss'
    print(f'Taking {loss_str} of >= {loss_cutoff} as \'impossible\'')
    print('Unconstructible molecules:')
    counts = bad_mols.value_counts()
    print(f'{counts[True]} / {counts.sum()}')
    return ~bad_mols

def plot_losses_together(chemspace,
                         chemspace_list,
                         chemspace_names,
                         chemspace_colours,
                         loss_cols=['length_loss', 'angle_loss', 'energy_loss'],
                         max_cutoffs=[1e2, 1e1, 1e3],
                         min_cutoffs=[1e-6, 1e-3, 3e-2],
                         use_logscale=[True, True, True],
                         mol_size_limit=None):
    '''
    Make side-by-side histogram plots of multiple chemical spaces' losses.
    Each plot contains one type of 'loss' value specified in `loss_cols`, and the
    lists `max_cutoffs`, `min_cutoffs`, `use_logscale` each specifies the options
    used for each of the `loss_cols` elements (and so must have the same length
    as `loss_cols`).
    Each chemical space in `chemspace_list` gets its own display name in
    `chemspace_names`, with colour `chemspace_colours`, and thus these will have
    the same length as `chemspace_list`.
    Inputs:
        chemspace: dictionary of chemical spaces, indexed by strings using labels
            from `chemspace_list`. Each chemical space is again a dictionary with
            entries ['df'] denoting the dataframe, and ['mols'] denoting the RDKit
            molecules sequence.
        chemspace_list: list of string, (internal) names of `chemspace` used
            for the plot
        chemspace_names: list of string, display latex names used in the plot for
            each of the spaces in `chemspace_list`. Same length as `chemsapce_list`.
        chemspace_colours: list of string, each specifying a colour, and corresponds
            to the colours used for each of the chemspaces in the plot. Same length
            as `chemspace_list`.
        loss_cols: list of strings, each an index of the 'loss' column used for the
            plot
        max_cutoffs, min_cutoffs: list of numbers, max/min cutoffs on the plot
            respectively for each of the loss columns. Same lengths as `loss_cols`.
        use_logscale: list of booleans, whether to use a logscale for the plot, for
            each of the `loss_cols` respectively. Same length as `loss_cols`.
        mol_size_limit: integer or None, only molecules with <= `mol_size_limit`
            heavy atoms are used for the plot. None to include all molecules.
    '''
    if not (len(chemspace_list) == len(chemspace_names) == len(chemspace_colours)):
        raise ValueError('lengths of chemspace_list, chemspace_names and chemspace_colours'
                         ' are not all equal.')
    if not (len(loss_cols) == len(max_cutoffs) == len(min_cutoffs) == len(use_logscale)):
        raise ValueError('lengths of loss_cols, max_cutoffs, use_logscale and min_cutoffs'
                         ' are not all equal.')

    if mol_size_limit:
        small_mols_location = {}
        for name in chemspace_list:
            small_mols_location[name] = [i for i, mol in enumerate(chemspace[name]['mols'])
                                         if mol and mol.GetNumHeavyAtoms() <= mol_size_limit]
            print(f'{name} small molecules: {len(small_mols_location[name])} '
                  f'(<= {mol_size_limit})')

    # plot the losses w.r.t. lengths and angles together
    n_plots = len(loss_cols)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(10,5*n_plots))
    for ax, losses_key, max_cutoff, min_cutoff, logscale in zip(
        axes, loss_cols, max_cutoffs, min_cutoffs, use_logscale):
        # Find the overall min/max losses over all of the chemical spaces
        losses_data = []
        for name in chemspace_list:
            df = chemspace[name]['df']
            if mol_size_limit:
                df = df.iloc[small_mols_location[name]]

            loss = df[losses_key].fillna(max_cutoff).clip(upper=max_cutoff, lower=min_cutoff) # avoid wide bar at 0
            losses_data.append(loss)

        min_loss, max_loss = min([min(losses) for losses in losses_data]), max([max(losses) for losses in losses_data])

        if logscale: # generating a log scale
            bins = np.geomspace(min_loss*0.99, max_loss*1.01, 30)
            ax.set_xscale('log')
        else:
            bins = np.linspace(min_loss, max_loss, 30)

        n, bins, patches = ax.hist(losses_data,
            weights=[np.ones_like(data)/len(data) for data in losses_data], # for outputting a pdf
            histtype='bar', color=chemspace_colours,
            bins=bins)

        ax.set_xlabel(f'{losses_key}')
        ax.set_ylabel('Density')

    plt.legend(patches, chemspace_names)
    plt.tight_layout()

# Evaluation methods
classification_metrics = {
    'accuracy': metrics.accuracy_score,
    'adjusted balanced accuracy': (lambda y_true, y_pred: metrics.balanced_accuracy_score(y_true, y_pred, adjusted=True)),
    'balanced accuracy': metrics.balanced_accuracy_score,
    'average precision': metrics.average_precision_score,
    #'negative logloss': metrics.log_loss,
    'f1 score': metrics.f1_score,
    'precision': metrics.precision_score,
    'recall': metrics.recall_score,
    'roc': metrics.roc_auc_score,
    'confusion matrix\n': metrics.confusion_matrix
}

def evaluate_classifier(model, X_test, y_test):
    print('Testing results:')
    y_pred = model.predict(X_test)
    for name, metric in classification_metrics.items():
        loss = metric(y_test, y_pred)
        print(f'{name} = {loss}')

def evaluate_classifier_and_get_results(model, X_test, y_test):
    results = {}
    y_pred = model.predict(X_test)
    for name, metric in classification_metrics.items():
        loss = metric(y_test, y_pred)
        results[name] = loss

    return results

def cross_validate_classifier(model, X_train, y_train, cv=5, verbose=5, n_jobs=7):
    # for use in model_selection.cross_validate
    classification_scorers = {
        name: metrics.make_scorer(metric)
        for name, metric in classification_metrics.items()
        if name not in ['confusion matrix\n']
    }

    print('Cross validation results:')
    scores = model_selection.cross_validate(model, X_train, y_train,
                                            cv=cv, scoring=classification_scorers,
                                            verbose=verbose, n_jobs=n_jobs)
    score_results = {name: (vals.mean(), vals.std()*2)
                     for name, vals in scores.items()}

    for name, results in score_results.items():
        mean_score, unc = results
        print(f'{name} = {mean_score} +- {unc}')

def describe_losses(db, db_name, losses_col='energy_loss', ignore_nans=True):
    losses = db[losses_col].astype(np.float64)
    if ignore_nans:
        nans_ids = losses.isna()
        losses = losses[~nans_ids]

    print(f'{db_name} losses distribution:')
    print(losses.describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]), '\n')

def avg_corr(idx, corr):
    '''
    Returns the average absolute value of correlation of `idx` with other features in
    `corr`.
    '''
    return corr.iloc[idx].abs().mean()
