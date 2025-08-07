# CLI Reference

TopoGen provides a command-line interface for topology generation operations.

## Usage

```bash
topogen [OPTIONS]
```

## Options

### Global Options

- `--config PATH`: Path to configuration file (default: config.yml)
- `-v, --verbose`: Enable verbose debug output
- `--help`: Show help message

### Examples

```bash
# Basic topology generation
topogen

# Use custom configuration
topogen --config scenarios/metro.yml

# Enable debug logging
topogen -v --config my-config.yml
```

## Configuration Files

TopoGen uses YAML configuration files to define topology parameters. Check the project repository for example configurations.
