### Code in this file are adapted from scikti-learn docs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import seaborn as sns


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ax1=None, ylim=None, cv=None,
                        scoring=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if ax1 is None:
        _, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    ax1.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    ax1.set_xlabel("Training examples")
    ax1.set_ylabel(scoring)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, scoring=scoring,
                       n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.subplots_adjust(right=.85)

    ax1.grid()
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    t_score = ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    cv_score = ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax1.set_ylabel('Negative Log Loss')

    # Plot n_samples vs fit_times
    ax2 = ax1.twinx()
    #ax2.grid()
    times = ax2.plot(train_sizes, fit_times_mean, ':', label='Fit Times (right axis)')
    ax2.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    ax2.set_xlabel("Training examples")
    ax2.set_ylabel("Fit Times (seconds)")
    #ax2.set_title("Scalability of the model")
    lines = t_score + cv_score + times
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='lower right')

    # Don't need this view
    # Plot fit_time vs score
    #axes[2].grid()
    #axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    #axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                     test_scores_mean + test_scores_std, alpha=0.1)
    #axes[2].set_xlabel("fit_times")
    #axes[2].set_ylabel("Score")
    #axes[2].set_title("Performance of the model")

    return plt
 

#https://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html
def plot_probability(
        X,
        y,
        clf,
        idx,
        num_of_num_cols,
        ax,
        ss,
        scaler=1.0,
        scatter=True):

    def inverse_transform(a, i):
        return a * ss.scale_[i] + ss.mean_[i]


    prob_alpha = 0.45
    x_min = X[:, idx[0]].min()*scaler
    x_max = X[:, idx[0]].max()*scaler
    y_min = X[:, idx[1]].min()*scaler
    y_max = X[:, idx[1]].max()*scaler

    cols = [i for i in range(X.shape[1]) if i not in idx]
    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100).T
    xx, yy = np.meshgrid(xx, yy)
    xfull = np.c_[xx.ravel(), yy.ravel()]
    plot_X = np.zeros((xfull.shape[0], X.shape[1]))
    plot_X[:, idx[0]] = xfull[:, 0]
    plot_X[:, idx[1]] = xfull[:, 1]
    plot_X[:, cols] = np.mean(X[:, cols], axis=0)

    probas = clf.predict_proba(plot_X)

    mask = np.ones(X.shape[0])
    width = 3
    while mask.sum() > 900 and width > .4:
        mask = np.ones(X.shape[0])

        # loop through numeric columns and get points within .5 of average
        for i in range(num_of_num_cols):
            if i not in idx:
                new_mask = mask * ((X[:, i] > - width) & (X[:, i] < width))
                if new_mask.sum() < 900:
                    break
                else:
                    mask = new_mask
        width -= 0.5        

    mask = mask.astype(bool)
    # if more than 100, then sample
    if mask.sum() > 900:
        indices = np.where(mask)[0]
        #print(indices)
        np.random.seed(0)
        new_indices = np.random.choice(np.array(indices), 1000)
        new_mask = np.zeros(mask.size)
        new_mask[new_indices] = 1
        mask = new_mask.astype(bool)

    n_classes = np.unique(y).size
    markers = 'xo'
    colors = ['purple', 'yellow']

    for k, label in enumerate(np.unique(y)):
        #print('here')
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(inverse_transform(x_min, 0),
                                           inverse_transform(x_max, 0),
                                           inverse_transform(y_min, 1),
                                           inverse_transform(y_max, 1)
                                           ), origin='lower',
                                   vmin=0, vmax=1,
                                   alpha=prob_alpha)
        #print(mask)
        #plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
        #print(X[mask])
        #print(imshow_handle)
        plot_mask = mask * (y == label)
        plot_mask = plot_mask.astype(bool)

        if scatter:
            if mask.any():
                plt.scatter(inverse_transform(X[plot_mask, 0], 0),
                            inverse_transform(X[plot_mask, 1], 1),
                            marker=markers[k],
                            c=colors[k],  #'gray',
                            #colormap='viridis',
                            edgecolor='k', alpha=0.8)
    #print(imshow_handle)
    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.colorbar(imshow_handle, cax=ax, alpha=prob_alpha, orientation='horizontal')

    return plt

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py    
def cont_hp_plot(df, metric, var, dtype, logx=False):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    df_var = 'param_model__' + var
    df_train_mean = 'mean_train_' + metric
    df_train_std = 'std_train_' + metric
    df_test_mean = 'mean_test_' + metric
    df_test_std = 'std_test_' + metric

    param_range = df[df_var].astype(dtype)
    
    train_scores_mean = df[df_train_mean]
    train_scores_std = df[df_train_std]
    test_scores_mean = df[df_test_mean]
    test_scores_std = df[df_test_std]
    lw = 2

    if logx:
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="C0", lw=lw)
    else:
        plt.plot(param_range, train_scores_mean, label="Training score",
                         color="C0", lw=lw)
    plt.fill_between(param_range,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.2,
                     color="C0", lw=lw)

    if logx:
        plt.semilogx(param_range, test_scores_mean, label="Validation score",
                         color="C1", lw=lw)
    else:
        plt.plot(param_range, test_scores_mean, label="Validation score",
                         color="C1", lw=lw)
    plt.fill_between(param_range,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.2,
                     color="C1", lw=lw)

    plt.legend(loc="best")
    plt.xlabel(var)
    plt.ylabel(metric)

    return plt

# adapted from here: https://towardsdatascience.com/using-3d-visualizations-to-tune-hyperparameters-of-ml-models-with-python-ba2885eab2e9
def contcont_hp_plot(df, metric, var1, var2):
    plt.clf()
    #fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig, axes = plt.subplots(nrows=2, figsize=(6, 8))
    #plt.subplots_adjust(bottom=.25, top=.8)
    #plt.subplots_adjust(top=.7)
    df_var1 = 'param_model__' + var1
    df_var2 = 'param_model__' + var2
    df_train_mean = 'mean_train_' + metric
    df_test_mean = 'mean_test_' + metric
    vmin = min([df[df_train_mean].min(),
                df[df_test_mean].min()])
    vmax= max([df[df_train_mean].max(),
                df[df_test_mean].max()])
    train_df = (df[[df_var1, df_var2, df_train_mean]].set_index(
        [df_var2, df_var1]).unstack()[df_train_mean])
    test_df = (df[[df_var1, df_var2, df_test_mean]].set_index(
        [df_var2, df_var1]).unstack()[df_test_mean])
    print('train_df in heatmap')
    print(train_df)
    sns.heatmap(train_df, ax=axes[0], vmin=vmin, vmax=vmax, annot=True, fmt='.3g')
    axes[0].set_title('Training')
    axes[0].set_xlabel(var1)
    axes[0].set_ylabel(var2)
    sns.heatmap(test_df, ax=axes[1], vmin=vmin, vmax=vmax, annot=True, fmt='.3g')
    axes[1].set_title('Validation')
    axes[1].set_xlabel(var1)
    axes[1].set_ylabel(var2)
    return plt 


def plot_epoch_curve(model):
    plt.clf()
    plt.plot(model.loss_curve_)
    plt.xlabel('epoch')
    plt.ylabel('log loss')
    return plt
