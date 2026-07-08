"""Profiling & metrics package (shared by the serving dashboard and benchmark)."""

from .metrics import MetricsCollector, RequestRecord

__all__ = ["MetricsCollector", "RequestRecord"]
