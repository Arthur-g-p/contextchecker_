"""Evaluation layer — measures pipeline quality against ground truth.

Three isolation levels:
  - ExtractorEvaluator: triplet-level extraction quality
  - CheckerEvaluator:   triplet-level entailment classification
  - MetaEvaluator:      item-level end-to-end factuality
"""
