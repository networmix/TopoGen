"""Standalone report generation for NetGraph analysis results.

Generates Jupyter notebooks and HTML reports from results.json files.
Separate from workflow execution to allow independent report generation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import nbformat

from ngraph.logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generate analysis reports from NetGraph results files.

    Creates Jupyter notebooks with analysis code and can optionally export to HTML.
    Uses the analysis registry to determine which analysis modules to run for each workflow step.
    """

    def __init__(self, results_path: Path = Path("results.json")):
        """Initialize report generator.

        Args:
            results_path: Path to results.json file containing analysis data.
        """
        self.results_path = results_path
        self._results: Dict[str, Any] = {}
        self._workflow_metadata: Dict[str, Any] = {}

    def load_results(self) -> None:
        """Load results from JSON file.

        Raises:
            FileNotFoundError: If results file doesn't exist.
            ValueError: If results file is invalid or empty.
        """
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        try:
            with open(self.results_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in results file: {e}") from e

        if not data:
            raise ValueError("Results file is empty")

        self._results = data
        self._workflow_metadata = data.get("workflow", {})

        # Check if we have any actual results (beyond just workflow metadata)
        if not any(k != "workflow" for k in data.keys()):
            raise ValueError(
                "No analysis results found in file (only workflow metadata)"
            )

        logger.info(
            f"Loaded results with {len(self._workflow_metadata)} workflow steps"
        )

    def generate_notebook(self, output_path: Path = Path("analysis.ipynb")) -> Path:
        """Generate Jupyter notebook with analysis code.

        Args:
            output_path: Where to save the notebook file.

        Returns:
            Path to the generated notebook file.

        Raises:
            ValueError: If no results are loaded.
        """
        if not self._results:
            raise ValueError("No results loaded. Call load_results() first.")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        notebook = self._create_analysis_notebook()

        with open(output_path, "w") as f:
            nbformat.write(notebook, f)

        logger.info(f"Notebook saved to: {output_path}")
        return output_path

    def generate_html_report(
        self,
        notebook_path: Path = Path("analysis.ipynb"),
        html_path: Path = Path("analysis_report.html"),
        include_code: bool = False,
    ) -> Path:
        """Generate HTML report from notebook.

        Args:
            notebook_path: Path to notebook file (will be created if doesn't exist).
            html_path: Where to save the HTML report.
            include_code: Whether to include code cells in HTML output.

        Returns:
            Path to the generated HTML file.

        Raises:
            RuntimeError: If nbconvert fails.
        """
        # Generate notebook if it doesn't exist
        if not notebook_path.exists():
            self.generate_notebook(notebook_path)

        # Ensure output directory exists
        html_path.parent.mkdir(parents=True, exist_ok=True)

        # Build nbconvert command
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "html",
            str(notebook_path),
            "--output",
            str(html_path),
        ]

        # Add --no-input flag to exclude code cells
        if not include_code:
            cmd.append("--no-input")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"HTML report saved to: {html_path}")
            return html_path
        except subprocess.CalledProcessError as e:
            logger.error(f"nbconvert failed: {e.stderr}")
            raise RuntimeError(f"Failed to generate HTML report: {e.stderr}") from e

    def _create_analysis_notebook(self) -> nbformat.NotebookNode:
        """Create notebook with analysis code based on loaded results."""
        nb = nbformat.v4.new_notebook()

        # Title cell
        title_cell = nbformat.v4.new_markdown_cell("# NetGraph Results Analysis")
        nb.cells.append(title_cell)

        # Setup cell
        setup_cell = self._create_setup_cell()
        nb.cells.append(setup_cell)

        # Data loading cell
        data_loading_cell = self._create_data_loading_cell()
        nb.cells.append(data_loading_cell)

        # Analysis overview cell
        overview_cell = self._create_analysis_overview_cell()
        nb.cells.append(overview_cell)

        # Generate analysis sections for each workflow step
        self._add_analysis_sections(nb)

        return nb

    def _create_setup_cell(self) -> nbformat.NotebookNode:
        """Create setup cell with imports and environment configuration."""
        setup_code = """# Setup analysis environment
from ngraph.workflow.analysis import (
    CapacityMatrixAnalyzer,
    SummaryAnalyzer,
    PackageManager,
    DataLoader,
    get_default_registry
)

# Setup packages and environment
package_manager = PackageManager()
setup_result = package_manager.setup_environment()

if setup_result['status'] != 'success':
    print("⚠️ Setup warning:", setup_result['message'])
else:
    print("✅ Environment setup complete")

# Initialize analysis registry
registry = get_default_registry()
print(f"Analysis registry loaded with {len(registry.get_all_step_types())} step types")"""

        return nbformat.v4.new_code_cell(setup_code)

    def _create_data_loading_cell(self) -> nbformat.NotebookNode:
        """Create data loading cell."""
        data_loading_code = f"""# Load analysis results
loader = DataLoader()
load_result = loader.load_results('{self.results_path.name}')

if load_result['success']:
    results = load_result['results']
    workflow_metadata = results.get('workflow', {{}})
    print(f"✅ Loaded {{len(results)-1}} analysis steps from {self.results_path.name}")
    print(f"Workflow contains {{len(workflow_metadata)}} steps")
else:
    print("❌ Load failed:", load_result['message'])
    results = {{}}
    workflow_metadata = {{}}"""

        return nbformat.v4.new_code_cell(data_loading_code)

    def _create_analysis_overview_cell(self) -> nbformat.NotebookNode:
        """Create analysis overview cell showing planned analysis steps."""
        overview_code = """# Analysis Overview
print("Analysis Plan")
print("=" * 60)

if 'workflow' in results and workflow_metadata:
    step_order = sorted(
        workflow_metadata.keys(),
        key=lambda step: workflow_metadata[step]["execution_order"]
    )

    for i, step_name in enumerate(step_order, 1):
        step_meta = workflow_metadata[step_name]
        step_type = step_meta["step_type"]

        analyses = registry.get_analyses(step_type)

        print(f"{i:2d}. {step_name} ({step_type})")

        if analyses:
            for analysis_config in analyses:
                analyzer_name = analysis_config.analyzer_class.__name__
                method_name = analysis_config.method_name
                print(f"    -> {analyzer_name}.{method_name}")
        else:
            print("    -> No analysis modules configured")

        # Check if data exists
        if step_name not in results:
            print("    ⚠️ No data found for this step")

        print()

    print(f"Total: {len(step_order)} workflow steps")
else:
    print("❌ No workflow metadata found")"""

        return nbformat.v4.new_code_cell(overview_code)

    def _add_analysis_sections(self, nb: nbformat.NotebookNode) -> None:
        """Add analysis sections for each workflow step."""
        if not self._workflow_metadata:
            return

        # Import analysis registry
        from ngraph.workflow.analysis import get_default_registry

        registry = get_default_registry()

        # Sort steps by execution order
        step_order = sorted(
            self._workflow_metadata.keys(),
            key=lambda step: self._workflow_metadata[step]["execution_order"],
        )

        for step_name in step_order:
            step_meta = self._workflow_metadata[step_name]
            step_type = step_meta["step_type"]

            # Add section header
            section_header = f"## {step_name} ({step_type})"
            nb.cells.append(nbformat.v4.new_markdown_cell(section_header))

            # Get registered analyses for this step type
            analyses = registry.get_analyses(step_type)

            if not analyses:
                # No analyses configured for this step type
                no_analysis_cell = nbformat.v4.new_code_cell(
                    f'print("INFO: No analysis modules configured for step type: {step_type}")'
                )
                nb.cells.append(no_analysis_cell)
                continue

            # Add analysis subsections
            for analysis_config in analyses:
                if len(analyses) > 1:
                    # Add subsection header if multiple analyses
                    subsection_header = f"### {analysis_config.section_title}"
                    nb.cells.append(nbformat.v4.new_markdown_cell(subsection_header))

                # Create analysis cell
                analysis_cell = self._create_analysis_cell(step_name, analysis_config)
                nb.cells.append(analysis_cell)

    def _create_analysis_cell(
        self, step_name: str, analysis_config
    ) -> nbformat.NotebookNode:
        """Create analysis code cell for specific step and analysis configuration."""
        analyzer_class_name = analysis_config.analyzer_class.__name__
        method_name = analysis_config.method_name
        section_title = analysis_config.section_title

        # Build kwargs for the analysis method
        kwargs_parts = [f"step_name='{step_name}'"]
        if analysis_config.kwargs:
            for key, value in analysis_config.kwargs.items():
                if isinstance(value, str):
                    kwargs_parts.append(f"{key}='{value}'")
                else:
                    kwargs_parts.append(f"{key}={value}")

        kwargs_str = ", ".join(kwargs_parts)

        analysis_code = f"""# {section_title}
if '{step_name}' in results:
    analyzer = {analyzer_class_name}()
    try:
        analyzer.{method_name}(results, {kwargs_str})
    except Exception as e:
        print(f"❌ Analysis failed: {{e}}")
else:
    print("❌ No data available for step: {step_name}")"""

        return nbformat.v4.new_code_cell(analysis_code)
