from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Context:
    name: str
    services: dict[str, Any]
    disposers: list[Callable[[], None]] = field(default_factory=list)

    def provide(self, key: str, value: Any) -> None:
        self.services[key] = value

    def get(self, key: str) -> Any:
        return self.services[key]

    def on_dispose(self, fn: Callable[[], None]) -> None:
        self.disposers.append(fn)

    def dispose(self) -> None:
        for fn in reversed(self.disposers):
            fn()
        self.disposers.clear()


@dataclass
class PluginManifest:
    name: str
    version: str
    api_version: str = "1"
    requires: list[str] = field(default_factory=list)
    contributes: list[str] = field(default_factory=list)


class PluginHost:
    def __init__(self) -> None:
        self.plugins: dict[str, Context] = {}
        self.adapters: dict[str, Any] = {}
        self.tools: dict[str, Any] = {}
        self.estimators: dict[str, Any] = {}

    def load(self, name: str, setup: Callable[[Context], None], services: dict[str, Any] | None = None) -> None:
        ctx = Context(name=name, services=dict(services or {}))
        setup(ctx)
        self.plugins[name] = ctx

    def unload(self, name: str) -> None:
        ctx = self.plugins.pop(name, None)
        if ctx:
            ctx.dispose()
