"""Base classes for workflow automation.

Defines the workflow step abstraction, registration decorator, and execution
wrapper that adds timing and logging. Steps implement `run()` and are executed
via `execute()` which records metadata and re-raises failures.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Type

from ngraph.logging import get_logger

if TYPE_CHECKING:
    # Only imported for type-checking; not at runtime, so no circular import occurs.
    from ngraph.scenario import Scenario

logger = get_logger(__name__)

# Registry for workflow step classes
WORKFLOW_STEP_REGISTRY: Dict[str, Type["WorkflowStep"]] = {}

# Global execution counter for tracking step order
_execution_counter = 0


def register_workflow_step(step_type: str):
    """Return a decorator that registers a `WorkflowStep` subclass.

    Args:
        step_type: Registry key used to instantiate steps from configuration.

    Returns:
        A class decorator that adds the class to `WORKFLOW_STEP_REGISTRY`.
    """

    def decorator(cls: Type["WorkflowStep"]) -> Type["WorkflowStep"]:
        WORKFLOW_STEP_REGISTRY[step_type] = cls
        return cls

    return decorator


@dataclass
class WorkflowStep(ABC):
    """Base class for all workflow steps.

    All workflow steps are automatically logged with execution timing information.
    All workflow steps support seeding for reproducible random operations.
    Workflow metadata is automatically stored in scenario.results for analysis.

    YAML Configuration:
        ```yaml
        workflow:
          - step_type: <StepTypeName>
            name: "optional_step_name"  # Optional: Custom name for this step instance
            seed: 42                    # Optional: Seed for reproducible random operations
            # ... step-specific parameters ...
        ```

    Attributes:
        name: Optional custom identifier for this workflow step instance,
            used for logging and result storage purposes.
        seed: Optional seed for reproducible random operations. If None,
            random operations will be non-deterministic.
    """

    name: str = ""
    seed: Optional[int] = None

    def execute(self, scenario: "Scenario") -> None:
        """Execute the workflow step with logging and metadata storage.

        This method wraps the abstract run() method with timing, logging, and
        automatic metadata storage for the analysis registry system.

        Args:
            scenario: The scenario to execute the step on.

        Returns:
            None

        Raises:
            Exception: Re-raises any exception raised by `run()` after logging
                duration and context.
        """
        global _execution_counter

        step_type = self.__class__.__name__
        step_name = self.name or step_type

        # Store workflow metadata before execution
        scenario.results.put_step_metadata(
            step_name=step_name, step_type=step_type, execution_order=_execution_counter
        )
        _execution_counter += 1

        logger.info(f"Starting workflow step: {step_name} ({step_type})")
        start_time = time.time()

        try:
            self.run(scenario)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(
                f"Completed workflow step: {step_name} ({step_type}) "
                f"in {duration:.3f} seconds"
            )
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(
                f"Failed workflow step: {step_name} ({step_type}) "
                f"after {duration:.3f} seconds - {type(e).__name__}: {e}"
            )
            raise

    @abstractmethod
    def run(self, scenario: "Scenario") -> None:
        """Execute the workflow step logic.

        This method should be implemented by concrete workflow step classes.
        It is called by execute() which handles logging, timing, and metadata storage.

        Args:
            scenario: The scenario to execute the step on.

        Returns:
            None
        """
        pass
