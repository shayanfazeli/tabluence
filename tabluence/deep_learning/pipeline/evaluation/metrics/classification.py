from typing import Dict, Any, List
import sklearn.metrics
import numpy


def compute_all_classification_metrics(
        epoch_y: numpy.ndarray,
        epoch_y_hat: numpy.ndarray,
        epoch_y_score: numpy.ndarray,
        labels: numpy.ndarray
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    epoch_y : numpy.ndarray
        The true labels for the epoch.
    epoch_y_hat : numpy.ndarray
        The predicted labels for the epoch.
    labels : numpy.ndarray
        The labels for the dataset.

    Returns
    -------
    `Dict[str, Any]`: the computed metrics, including the following keys:
        - accuracy
        - accuracy details
        - precision (included in the `prf` key)
        - recall (included in the `prf` key)
        - f1 (included in the `prf` key)
        - classification_report
    """
    stats = dict()
    stats['accuracy'] = sklearn.metrics.accuracy_score(epoch_y, epoch_y_hat, normalize=True)
    stats['accuracy_details'] = {
        'number_of_correctly_classified': sklearn.metrics.accuracy_score(epoch_y, epoch_y_hat, normalize=False),
        'total_number_of_samples': epoch_y.shape[0]
    }
    epoch_y_score = epoch_y_score[numpy.arange(epoch_y_score.shape[0]), epoch_y]
    stats['roc_auc'] = {
        x: sklearn.metrics.roc_auc_score(y_true=epoch_y, y_score=epoch_y_score, average=x)
        for x in ['macro', 'micro', 'samples']}

    stats['classification_report'] = sklearn.metrics.classification_report(
        y_true=epoch_y,
        y_pred=epoch_y_hat,
        labels=labels,
        output_dict=True,
        zero_division=0)

    stats['confusion_matrix'] = {
        'normalized': {k: sklearn.metrics.confusion_matrix(
            y_true=epoch_y,
            y_pred=epoch_y_hat,
            labels=labels, normalize=k) for k in ['true', 'pred', 'all']},
        'not_normalized': sklearn.metrics.confusion_matrix(y_true=epoch_y,
                                                               y_pred=epoch_y_hat,
                                                               labels=labels, normalize=None)
    }

    stats['prf'] = {
        x: sklearn.metrics.precision_recall_fscore_support(
            y_true=epoch_y,
            y_pred=epoch_y_hat,
            labels=labels,
            average=x,
            zero_division=0
        ) for x in ['macro', 'micro', None]}

    return stats
