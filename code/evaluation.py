from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


def evaluation(ground_truth, predicted_gesture):
    """
    Evaluates the model using various metrics.
    Args:
        ground_truth (np.ndarray): The ground truth labels.
        predicted_gesture (np.ndarray): The predicted labels.
    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    f1 = f1_score(ground_truth, predicted_gesture, average="weighted")
    precision = precision_score(ground_truth, predicted_gesture, average="weighted")
    recall = recall_score(ground_truth, predicted_gesture, average="weighted")
    top_k_accuracy = top_k_accuracy_score(ground_truth, predicted_gesture, k=3)
    roc_auc = roc_auc_score(ground_truth, predicted_gesture, average="weighted", multi_class="ovo")
    CLER = 1 - len(ground_truth[ground_truth == predicted_gesture]) / len(ground_truth)