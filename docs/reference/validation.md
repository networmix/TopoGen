## Validation subsystem

TopoGen performs two layers of validation for generated NetGraph scenarios:

- Intra-scenario checks on the raw YAML/dict (no `ngraph` dependency).
- Optional `ngraph`-powered schema and audit checks on the expanded network.

### API

```python
from topogen.validation import validate_scenario_yaml, validate_scenario_dict
```

- `validate_scenario_dict(data, ig_coords=None) -> list[str]`
  - Fast checks on the parsed YAML dict: metro PoP/DC attribute consistency, required DC attributes, workflow references to traffic matrices and failure policies, isolation hints, and DC adjacency capacity vs traffic demand.

- `validate_scenario_yaml(scenario_yaml, integrated_graph_path=None, run_ngraph=True) -> list[str]`
  - Parses YAML, runs `validate_scenario_dict`, optionally loads the integrated graph for metro coordinate cross-check, and then runs `ngraph` audits if enabled.

### Audits (ngraph)

The audit pipeline runs:

1. Schema and isolation via `ngraph.scenario.Scenario`.
2. Expansion checks: groups and scenario/blueprint adjacency must expand to nodes/links.
3. Node role presence.
4. Node hardware presence and basic validity against `components.hw_component` and component library.
5. Link optics mapping and blueprint hardware presence/capacity checks.
6. Platform port budget vs optics modules required.

All functions return human-readable issue strings. An empty list indicates no detected issues.

### Where validation runs

- The scenario pipeline (`topogen.scenario.assembly.build_scenario`) focuses on construction and does not validate.
- The CLI `build` command validates after writing the scenario YAML using `validate_scenario_yaml` with the integrated graph path and `run_ngraph=True`.
