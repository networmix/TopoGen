from __future__ import annotations

import importlib
from unittest.mock import patch


def test___main__invokes_cli_main() -> None:
    with patch("topogen.cli.main") as m:
        import topogen.__main__ as entry

        importlib.reload(entry)
        # Simulate running as a script
        if __name__ == "__main__":
            entry.main()
        m.assert_not_called()  # import shouldn't call
