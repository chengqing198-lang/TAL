"""Semantic error types and exit codes.

Exit codes are part of the interface: CI can distinguish "the experiment ran and
was rejected" from "a tool crashed" without parsing logs.
"""

EXIT_OK = 0
EXIT_GATE_FAILED = 10
EXIT_UPSTREAM_FAILED = 20
EXIT_USAGE = 30
EXIT_SPEC = 31
EXIT_BUDGET = 40
EXIT_LOCKED = 41
EXIT_INTERRUPTED = 50


class TalError(Exception):
    """Base class for all TAL errors."""

    exit_code = EXIT_USAGE


class SpecError(TalError):
    exit_code = EXIT_SPEC


class StageError(TalError):
    """A pipeline stage failed (missing artifact, internal or upstream crash)."""

    exit_code = EXIT_UPSTREAM_FAILED


class UpstreamError(TalError):
    """A wrapped external tool failed in a way we do not classify further."""

    exit_code = EXIT_UPSTREAM_FAILED


class GateFailed(TalError):
    """The experiment completed but did not satisfy the acceptance gates."""

    exit_code = EXIT_GATE_FAILED


class BudgetExceeded(TalError):
    exit_code = EXIT_BUDGET


class LockTimeout(TalError):
    exit_code = EXIT_LOCKED


class UserInterrupted(TalError):
    exit_code = EXIT_INTERRUPTED
