"""Shadow model poisoning for strategic classification.

Adapted from poisoning/shadow.py. Trains a shadow classifier on observed data,
identifies agents near the decision boundary, and shifts their strategic
features toward it. Only modifies strategic features.
"""
