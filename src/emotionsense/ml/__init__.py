"""Shared ML: preprocessing, feature extraction, and model definitions.

This package is imported by BOTH the training plane and the inference plane, which is
what guarantees feature/preprocess parity at serve time (ADR-3).
"""
