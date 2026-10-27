"""Incremental test dependency resolution without importing analyzed backends."""

from .catalog import OperationCatalog, OperationSpec
from .cli import TestRunRequest, TestRunner
from .dependencies import OperationDependencyResolver, OperationResolutionResult
from .policy import SelectionPolicy
from .selection import IncrementalSelectionResult, IncrementalTestSelector
from .validation import CatalogValidationResult, validate_catalog

__all__ = [
    "CatalogValidationResult",
    "IncrementalSelectionResult",
    "IncrementalTestSelector",
    "OperationCatalog",
    "OperationDependencyResolver",
    "OperationResolutionResult",
    "OperationSpec",
    "SelectionPolicy",
    "TestRunRequest",
    "TestRunner",
    "validate_catalog",
]
