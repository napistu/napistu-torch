"""Utilities for applying and decoding labels in napistu-torch."""

from typing import List, Optional, Union

import torch

from napistu_torch.labels.labeling_manager import LabelingManager


def decode_labels(
    encoded_labels: torch.Tensor,
    labeling_manager: LabelingManager,
    missing_value: int = -1,
) -> List[Optional[Union[str, int, float]]]:
    """Decode integer-encoded labels back to their original values.

    This function takes encoded labels (typically from a NapistuData.y tensor)
    and decodes them back to their original string/numeric values using the
    label_names mapping from a LabelingManager.

    Parameters
    ----------
    encoded_labels : torch.Tensor
        Tensor of integer-encoded labels (typically from NapistuData.y)
    labeling_manager : LabelingManager
        The labeling manager containing the label_names mapping
    missing_value : int, default=-1
        The integer value used to represent missing labels

    Returns
    -------
    List[Optional[Union[str, int, float]]]
        List of decoded labels, with None for missing values

    Examples
    --------
    >>> # Assuming we have encoded labels and a labeling manager
    >>> encoded_labels = torch.tensor([0, 1, 0, -1, 2])
    >>> decoded = decode_labels(encoded_labels, labeling_manager)
    >>> print(decoded)  # ['protein', 'metabolite', 'protein', None, 'drug']
    """
    if labeling_manager.label_names is None:
        raise ValueError(
            "LabelingManager must have label_names mapping to decode labels"
        )

    decoded_labels = []
    for label_int in encoded_labels:
        if label_int.item() == missing_value:
            # Handle missing values
            decoded_labels.append(None)
        elif label_int.item() in labeling_manager.label_names:
            decoded_labels.append(labeling_manager.label_names[label_int.item()])
        else:
            # Unknown label value - this shouldn't happen in normal operation
            raise ValueError(
                f"Unknown label value {label_int.item()} not found in label_names mapping. "
                f"Available labels: {list(labeling_manager.label_names.keys())}"
            )

    return decoded_labels
