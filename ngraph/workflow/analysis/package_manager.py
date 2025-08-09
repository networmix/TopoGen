"""Package management for notebook analysis components.

Provides light-weight helpers to ensure plotting/display packages are available
in interactive environments and to apply sensible defaults.
"""

from typing import Any, Dict

import itables.options as itables_opt
import matplotlib.pyplot as plt


class PackageManager:
    """Manage package installation and imports for notebooks."""

    REQUIRED_PACKAGES = {
        "itables": "itables",
        "matplotlib": "matplotlib",
    }

    @classmethod
    def check_and_install_packages(cls) -> Dict[str, Any]:
        """Check for required packages and install if missing.

        Returns:
            Status dictionary with installation results and messages.
        """
        import importlib
        import subprocess
        import sys

        missing_packages = []

        for package_name, pip_name in cls.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(package_name)
            except ImportError:
                missing_packages.append(pip_name)

        result = {
            "missing_packages": missing_packages,
            "installation_needed": len(missing_packages) > 0,
        }

        if missing_packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install"] + missing_packages
                )
                result["installation_success"] = True
                result["message"] = (
                    f"Successfully installed: {', '.join(missing_packages)}"
                )
            except subprocess.CalledProcessError as e:
                result["installation_success"] = False
                result["error"] = str(e)
                result["message"] = f"Installation failed: {e}"
        else:
            result["message"] = "All required packages are available"

        return result

    @classmethod
    def setup_environment(cls) -> Dict[str, Any]:
        """Set up the notebook environment.

        Returns:
            Status dictionary with environment configuration details.
        """
        # Check and install packages
        install_result = cls.check_and_install_packages()

        if not install_result.get("installation_success", True):
            return install_result

        try:
            # Configure matplotlib
            plt.style.use("seaborn-v0_8")

            # Configure itables
            itables_opt.lengthMenu = [10, 25, 50, 100, 500, -1]
            itables_opt.maxBytes = 10**7  # 10MB limit
            itables_opt.maxColumns = 200  # Allow more columns
            itables_opt.showIndex = True  # Always show DataFrame index as a column

            # Configure warnings
            import warnings

            warnings.filterwarnings("ignore")

            return {
                "status": "success",
                "message": "Environment setup complete",
                **install_result,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Environment setup failed: {str(e)}",
                **install_result,
            }
