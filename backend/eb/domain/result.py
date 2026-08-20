from __future__ import annotations

from pydantic import BaseModel, Field


class Violation(BaseModel):
    path: str
    expected: object
    actual: object
    hint: str = ""

    def __init__(self, path: str, expected: object, actual: object, hint: str = "") -> None:
        super().__init__(path=path, expected=expected, actual=actual, hint=hint)


class CheckResult(BaseModel):
    passed: bool
    violations: list[Violation] = Field(default_factory=list)

    @classmethod
    def ok(cls) -> CheckResult:
        return cls(passed=True)

    @classmethod
    def fail(cls, *violations: Violation) -> CheckResult:
        return cls(passed=False, violations=list(violations))
