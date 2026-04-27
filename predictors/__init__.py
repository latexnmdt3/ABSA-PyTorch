"""Predictor wrappers for each method in the ABSA survey.

Every file in this package implements the :class:`evaluate.Predictor`
protocol. Teammates only have to fill in the ``__init__`` (load the model)
and ``predict`` (produce a single ``(aspect, sentiment, raw_output)`` tuple)
methods. ``evaluate.py`` does everything else.
"""
