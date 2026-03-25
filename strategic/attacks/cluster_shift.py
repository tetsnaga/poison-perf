"""Cluster shift poisoning for strategic classification.

Adapted from poisoning/black_box.py (black_box_cluster). Shifts epsilon fraction
of each class's strategic features toward the opposing class mean to confuse
the decision boundary. Only modifies strategic features; labels unchanged.
"""
