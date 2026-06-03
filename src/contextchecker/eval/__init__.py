"""Evaluation layer — measures pipeline quality against ground truth.

Three isolation levels:
  - ExtractorEvaluator: triplet-level extraction quality
  - CheckerEvaluator:   triplet-level NLI classification
  - MetaEvaluator:      item-level end-to-end factuality
"""
