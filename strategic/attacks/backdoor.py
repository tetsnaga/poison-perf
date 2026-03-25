"""Backdoor poisoning for strategic classification.

Adversarial agents adopt a specific "trigger" pattern in their strategic features
during training, hoping the classifier learns a spurious correlation. At test time,
new agents can adopt the same trigger to get misclassified.

Unlike non-backdoor attacks, the goal is not to degrade overall model performance
but to create a hidden vulnerability exploitable by agents who know the trigger.
"""
