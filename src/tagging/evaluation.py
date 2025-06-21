def match_exact(example : str, ground_truth : str) -> bool:
    return example.lower() == ground_truth.lower()

def evaluate_example_exact(example : dict, ground_truth : dict) -> bool:
    return match_exact(example["label"], ground_truth["label"])

def evaluate_example_in_top(example : dict, ground_truth : dict, k=3) -> bool:
    soft_targets = [e["label"] for e in example["soft_targets"]][:k]
    return ground_truth["label"] in soft_targets


import collections


import collections

def calculate_top_k_accuracy(examples, ground_truths, k=3) -> float:
    """
    Calculates the top-k accuracy for a given dataset.

    This metric measures the proportion of examples where the predicted 'label'
    is present within the top-k 'soft_targets' of the corresponding ground truth.
    The matching between an example and a ground truth is done based on
    'paper_id' and 'level'.

    Args:
        examples (list[dict]): A list of dictionaries, where each dictionary
            represents a prediction and must contain 'paper_id', 'level', and 'label' keys.
        ground_truths (list[dict]): A list of dictionaries, where each dictionary
            represents the ground truth and must contain 'paper_id', 'level',
            and 'soft_targets' keys. The 'soft_targets' is a list of dicts,
            each with a 'label' and 'confidence'.
        k (int, optional): The number of top soft targets to consider for a match.
            Defaults to 3.

    Returns:
        float: The top-k accuracy, a value between 0.0 and 1.0. Returns 0.0
            if the examples list is empty.
    """
    # If there are no examples to evaluate, the accuracy is 0.
    if not examples:
        print("No examples to evaluate.")
        return 0.0

    # Create a lookup for ground truths for efficient access.
    # The key is a tuple of (paper_id, level) for unique identification.
    ground_truth_map = {
        (gt['paper_id'], gt['level']): gt for gt in ground_truths
    }

    correct_predictions = 0
    total_examples = len(examples)

    # Iterate through each example to evaluate its prediction.
    for example in examples:
        key = (example['paper_id'], example['level'])
        ground_truth = ground_truth_map.get(key)

        # A prediction can only be correct if a corresponding ground truth exists.
        if ground_truth:
            predicted_label = example['label']

            # Get the top-k labels from the soft targets of the ground truth.
            # Sort by confidence in descending order to get the top predictions.
            soft_targets = ground_truth.get('soft_targets', [])
            top_k_labels = [
                target['label'] for target in sorted(
                    soft_targets,
                    key=lambda x: x.get('confidence', 0),
                    reverse=True
                )[:k]
            ]

            # Check if the predicted label is in the top-k ground truth labels.
            if predicted_label in top_k_labels:
                correct_predictions += 1

    # Calculate the accuracy as the fraction of correct predictions.
    return correct_predictions / total_examples if total_examples > 0 else 0.0

def calculate_accuracy(examples, ground_truths) -> float:
    """
    Calculates the top-k accuracy for a given dataset.

    This metric measures the proportion of examples where the predicted 'label'
    is present within the top-k 'soft_targets' of the corresponding ground truth.
    The matching between an example and a ground truth is done based on
    'paper_id' and 'level'.

    Args:
        examples (list[dict]): A list of dictionaries, where each dictionary
            represents a prediction and must contain 'paper_id', 'level', and 'label' keys.
        ground_truths (list[dict]): A list of dictionaries, where each dictionary
            represents the ground truth and must contain 'paper_id', 'level',
            and 'soft_targets' keys. The 'soft_targets' is a list of dicts,
            each with a 'label' and 'confidence'.
        k (int, optional): The number of top soft targets to consider for a match.
            Defaults to 3.

    Returns:
        float: The top-k accuracy, a value between 0.0 and 1.0. Returns 0.0
            if the examples list is empty.
    """
    # If there are no examples to evaluate, the accuracy is 0.
    if not examples:
        print("No examples to evaluate.")
        return 0.0

    # Create a lookup for ground truths for efficient access.
    # The key is a tuple of (paper_id, level) for unique identification.
    ground_truth_map = {
        (gt['paper_id'], gt['level']): gt for gt in ground_truths
    }

    correct_predictions = 0
    total_examples = len(examples)

    # Iterate through each example to evaluate its prediction.
    for example in examples:
        key = (example['paper_id'], example['level'])
        ground_truth = ground_truth_map.get(key)

        # A prediction can only be correct if a corresponding ground truth exists.
        if ground_truth:
            predicted_label = example['label']

            # Get the top-k labels from the soft targets of the ground truth.
            # Sort by confidence in descending order to get the top predictions.
            ground_truth_label = ground_truth.get('soft_targets', "")

            # Check if the predicted label is in the top-k ground truth labels.
            if match_exact(example["label"], ground_truth_label):
                correct_predictions += 1

    # Calculate the accuracy as the fraction of correct predictions.
    return correct_predictions / total_examples if total_examples > 0 else 0.0


if __name__ == "__main__":
    # Example Usage:
    examples = [
        {
          "paper_id": 4216, "level": 1, "label": "Computing methodologies",
          "input": "...", "rationale":"...", "parent_path": []
        },
        {
          "paper_id": 1234, "level": 2, "label": "Data mining", # This is NOT in the top 2
          "input": "...", "rationale":"...", "parent_path": []
        },
        {
          "paper_id": 5678, "level": 1, "label": "Applied computing", # This one is correct
          "input": "...", "rationale":"...", "parent_path": []
        }
    ]

    ground_truths = [
        {
          "paper_id": 4216, "level": 1, "label": "Computing methodologies", "confidence": 0.41,
          "soft_targets": [
            {"label": "Computing methodologies", "confidence": 0.41, "rationale": "..."},
            {"label": "Human-centered computing", "confidence": 0.30, "rationale": "..."},
            {"label": "Applied computing", "confidence": 0.28, "rationale": "..."}
          ],
          "input": "...", "rationale":"...", "parent_path": []
        },
        {
          "paper_id": 1234, "level": 2, "label": "Artificial intelligence", "confidence": 0.5,
          "soft_targets": [
            {"label": "Artificial intelligence", "confidence": 0.5, "rationale": "..."},
            {"label": "Machine learning", "confidence": 0.4, "rationale": "..."},
            {"label": "Data mining", "confidence": 0.1, "rationale": "..."}
          ],
          "input": "...", "rationale":"...", "parent_path": []
        },
        {
          "paper_id": 5678, "level": 1, "label": "Applied computing", "confidence": 0.9,
          "soft_targets": [
            {"label": "Applied computing", "confidence": 0.9, "rationale": "..."},
            {"label": "Security and privacy", "confidence": 0.05, "rationale": "..."},
            {"label": "Software engineering", "confidence": 0.05, "rationale": "..."}
          ],
          "input": "...", "rationale":"...", "parent_path": []
        }
    ]

    # --- Test Cases ---

    # Test with k=3. Two predictions are correct (4216, 5678) and one is correct (1234 is 3rd).
    # Expected: 3/3 = 1.0
    accuracy_k3 = calculate_top_k_accuracy(examples, ground_truths, k=3)
    print(f"--- Test Case 1: Top-3 Accuracy ---")
    print(f"Evaluation result: {accuracy_k3:.4f}\n") # Expected: 1.0000

    # Test with k=2. Two predictions are correct (4216, 5678), but paper 1234's label ('Data mining') is 3rd.
    # Expected: 2/3 = 0.6667
    accuracy_k2 = calculate_top_k_accuracy(examples, ground_truths, k=2)
    print("--- Test Case 2: Top-2 Accuracy ---")
    print(f"Evaluation result: {accuracy_k2:.4f}\n") # Expected: 0.6667

    # Test with k=1. Two predictions are correct (4216, 5678).
    # Expected: 2/3 = 0.6667
    accuracy_k1 = calculate_top_k_accuracy(examples, ground_truths, k=1)
    print("--- Test Case 3: Top-1 Accuracy (Precision@1) ---")
    print(f"Evaluation result: {accuracy_k1:.4f}\n") # Expected: 0.6667

    # Test with an example that has no matching ground truth.
    examples_with_missing = examples + [{
        "paper_id": 9999, "level": 1, "label": "Some Label", # No ground truth for this
        "input": "...", "rationale":"...", "parent_path": []
    }]
    # Expected: 3 correct out of 4 total examples = 0.75
    accuracy_missing = calculate_top_k_accuracy(examples_with_missing, ground_truths, k=3)
    print("--- Test Case 4: With Missing Ground Truth ---")
    print(f"Evaluation result: {accuracy_missing:.4f}\n") # Expected: 0.7500