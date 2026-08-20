from __future__ import annotations

from eb.adapters.base import Adapter, RenderedJob
from eb.domain.job import JobSpec


class DataRetargetAdapter(Adapter):
    """Imported for the file map; canonical class lives in data_convert.py."""

    name = "data_retarget"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        from eb.adapters.data_convert import DataRetargetAdapter as Impl

        return Impl().render(spec, run_id)
