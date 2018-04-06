import numpy as np


def hamming_score(true_labels, predicts):
    """ hamming_score
    
    Computes de hamming score, which is known as the label-based accuracy,  
    designed for multi-label problems. It's defined as the number of correctly 
    predicted labels divided by all classified labels.
     
    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true labels for all the classification tasks and for 
        n_samples.
    
    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for 
        n_samples.
    
    Returns
    -------
    float
        The hamming score, or label-based accuracy, for the given sets.
    
    Examples
    --------
    >>> from skmultiflow.evaluation.metrics.metrics import hamming_score
    >>> true_labels = [[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,1,1]]
    >>> predictions = [[0,1,0,1],[0,1,1,0],[0,1,0,1],[1,1,1,1]]
    >>> hamming_score(true_labels, predictions)
    0.75
    
    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    return np.sum((true_labels == predicts) * 1.) / N / L

def j_index(true_labels, predicts):
    """ j_index
    
    Computes the Jaccard Index of the given set, which is also called the 
    'intersection over union' in multi-label settings. It's defined as the 
    intersection between the true label's set and the prediction's set, 
    divided by the sum, or union, of those two sets.
    
    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true labels for all the classification tasks and for 
        n_samples.
        
    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for 
        n_samples.
    
    Returns
    -------
    float
        The J-index, or 'intersection over union', for the given sets. 
    
    Examples
    --------
    >>> from skmultiflow.evaluation.metrics.metrics import j_index
    >>> true_labels = [[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,1,1]]
    >>> predictions = [[0,1,0,1],[0,1,1,0],[0,1,0,1],[1,1,1,1]]
    >>> j_index(true_labels, predictions)
    0.66666666666666663
    
    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    s = 0.0
    for i in range(N):
        inter = sum((true_labels[i, :] * predicts[i, :]) > 0) * 1.
        union = sum((true_labels[i, :] + predicts[i, :]) > 0) * 1.
        if union > 0:
            s += inter / union
        elif np.sum(true_labels[i, :]) == 0:
            s += 1.
    return s * 1. / N

def exact_match(true_labels, predicts):
    """ exact_match
    
    This is the most strict metric for the multi label setting. It's defined 
    as the percentage of samples that have all their labels correctly classified.
    
    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true labels for all the classification tasks and for 
        n_samples.
        
    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for 
        n_samples.
        
    Returns
    -------
    float
        The exact match percentage between the given sets.  
    
    Examples
    --------
    >>> from skmultiflow.evaluation.metrics.metrics import exact_match
    >>> true_labels = [[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,1,1]]
    >>> predictions = [[0,1,0,1],[0,1,1,0],[0,1,0,1],[1,1,1,1]]
    >>> exact_match(true_labels, predictions)
    0.5
    
    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    return np.sum(np.sum((true_labels == predicts) * 1, axis=1)==L) * 1. / N
