"""Utilities for creating pretty labels."""


def format_metric_label(text: str) -> str:
    """
    Format a label by replacing underscores with spaces, title casing,
    and ensuring AUC is always uppercase.

    Parameters
    ----------
    text : str
        Input text (e.g., 'val_auc', 'test_auc')

    Returns
    -------
    str
        Formatted label (e.g., 'Val AUC', 'Test AUC')

    Examples
    --------
    >>> format_label('val_auc')
    'Val AUC'
    >>> format_label('test_auc')
    'Test AUC'
    >>> format_label('relation_type')
    'Relation Type'
    """
    formatted = text.replace("_", " ").title().replace("Auc", "AUC")
    return formatted
