#!/bin/bash
# Setup script for developers

echo "🔧 Setting up development environment..."

# Create and activate virtual environment
if [ ! -d "topogen-venv" ]; then
    echo "🐍 Creating virtual environment 'topogen-venv'..."
    python -m venv topogen-venv
else
    echo "🐍 Virtual environment 'topogen-venv' already exists"
fi

echo "⚡ Activating virtual environment..."
source topogen-venv/bin/activate

# Install the package with dev dependencies
echo "📦 Installing package with dev dependencies..."
pip install -e '.[dev]'

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files to ensure everything is set up correctly
echo "✅ Running pre-commit checks..."
pre-commit run --all-files

echo "🎉 Development environment setup complete!"
echo ""
echo "🚀 You're ready to contribute! Pre-commit hooks will now run automatically on each commit."
echo "💡 To manually run all checks: pre-commit run --all-files"
echo "⚠️  Remember to activate the virtual environment in new terminal sessions: source topogen-venv/bin/activate"
