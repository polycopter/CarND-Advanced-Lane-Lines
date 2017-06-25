import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # Implement batching
    total = len(features)
    numbatches = total//batch_size
    check = numbatches * batch_size
    batchz = []
    for i in range(numbatches):
        batchz.append([features[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size]])
    if check < total:
        remainder = total - check
        ls = numbatches*batch_size
        batchz.append([features[ls:ls+batch_size], labels[ls:ls+batch_size]])

    return batchz
