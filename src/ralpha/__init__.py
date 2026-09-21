"""residual-alpha: an honest evaluation harness for time-series foundation
models on cross-sectional equity returns.

The pipeline forecasts BETA-RESIDUAL returns, scores every model against a
zero-forecast null, checks that predictive intervals are calibrated before
anything downstream consumes them, and reserves a holdout window that the
development commands physically refuse to read.
"""

__version__ = "0.1.0"
