from collections.abc import Iterable


def pairwise_f1(predicted: Iterable[int], expected: Iterable[int]) -> dict[str, float]:
    predicted, expected = list(predicted), list(expected)
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must have the same length")
    true_positive = false_positive = false_negative = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            same_predicted = predicted[left] == predicted[right] and predicted[left] >= 0
            same_expected = expected[left] == expected[right] and expected[left] >= 0
            true_positive += int(same_predicted and same_expected)
            false_positive += int(same_predicted and not same_expected)
            false_negative += int(not same_predicted and same_expected)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pairwise_precision": precision, "pairwise_recall": recall, "f1": f1}

