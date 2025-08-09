"""Data loading utilities for notebook analysis.

Provides simple JSON loading with basic validation and structured status output.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union


class DataLoader:
    """Handles loading and validation of analysis results."""

    @staticmethod
    def load_results(json_path: Union[str, Path]) -> Dict[str, Any]:
        """Load results from a JSON file with error handling.

        Args:
            json_path: Path to a JSON file containing analysis results.

        Returns:
            A dictionary with success flag, message, and parsed results.
        """
        json_path = Path(json_path)

        result = {
            "file_path": str(json_path),
            "success": False,
            "results": {},
            "message": "",
        }

        try:
            if not json_path.exists():
                result["message"] = f"Results file not found: {json_path}"
                return result

            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            if not isinstance(results, dict):
                result["message"] = "Invalid results format - expected dictionary"
                return result

            result.update(
                {
                    "success": True,
                    "results": results,
                    "message": f"Loaded {len(results):,} analysis steps from {json_path.name}",
                    "step_count": len(results),
                    "step_names": list(results.keys()),
                }
            )

        except json.JSONDecodeError as e:
            result["message"] = f"Invalid JSON format: {str(e)}"
        except Exception as e:
            result["message"] = f"Error loading results: {str(e)}"

        return result
