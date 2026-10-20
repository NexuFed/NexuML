"""Trusted CLI scenario file; reusable components stay in the installed library."""

from agent_demo.scenarios import regression

HYPOTHESIS = "The external library completes a CPU lifecycle without final-test data."
TAGS = ["smoke", "research-only"]


def scenario():
    """Return the installed library's smoke scenario."""
    return regression()
