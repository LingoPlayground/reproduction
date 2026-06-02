"""Shared data models used across pipeline stages.

CutPoint lives here so that scene_detection (Stage 1b) and
timeline_plan (Stage 3) can both import it without creating
a reverse dependency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CutPoint:
    """A detected scene cut boundary in seconds."""
    time_sec: float
