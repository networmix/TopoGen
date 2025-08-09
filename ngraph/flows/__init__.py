"""Flow primitives and routing policies.

This subpackage defines the building blocks for demand routing:

- FlowIndex: Immutable identifier for a flow.
- Flow: Routed demand portion bound to a `PathBundle`.
- FlowPolicy: Creates, places, rebalances, and removes flows on a
  `StrictMultiDiGraph`.
- FlowPolicyConfig and get_flow_policy(): Factory for common policy presets.

Components here interact with `ngraph.algorithms` for path selection and
placement, and with `ngraph.paths` for path bundle representation.
"""
