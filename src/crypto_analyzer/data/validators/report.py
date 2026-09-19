"""Structured validation results that preserve all detected issues."""

from dataclasses import dataclass
from enum import StrEnum

from crypto_analyzer.data.exceptions import DataValidationError


class IssueSeverity(StrEnum):
    """Severity assigned to a validation issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable problem found without changing the source dataset."""

    severity: IssueSeverity
    code: str
    message: str
    rows: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable collection of errors and warnings for one dataset."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is IssueSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is IssueSeverity.WARNING
        )

    @property
    def is_valid(self) -> bool:
        """Whether the dataset has no validation errors."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise one clear exception summarizing all validation errors."""
        if not self.errors:
            return
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
        raise DataValidationError(summary)
