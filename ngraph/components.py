"""Component and ComponentsLibrary classes for hardware cost modeling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

from ngraph.yaml_utils import normalize_yaml_dict_keys


@dataclass
class Component:
    """A generic component that can represent chassis, line cards, optics, etc.
    Components can have nested children, each with their own cost, power, etc.

    Attributes:
        name (str): Name of the component (e.g., "SpineChassis" or "400G-LR4").
        component_type (str): A string label (e.g., "chassis", "linecard", "optic").
        description (str): A human-readable description of this component.
        cost (float): Cost (capex) of a single instance of this component.
        power_watts (float): Typical/nominal power usage (watts) for one instance.
        power_watts_max (float): Maximum/peak power usage (watts) for one instance.
        capacity (float): A generic capacity measure (e.g., platform capacity).
        ports (int): Number of ports if relevant for this component.
        count (int): How many identical copies of this component are present.
        attrs (Dict[str, Any]): Arbitrary key-value attributes for extra metadata.
        children (Dict[str, Component]): Nested child components (e.g., line cards
            inside a chassis), keyed by child name.
    """

    name: str
    component_type: str = "generic"
    description: str = ""
    cost: float = 0.0

    power_watts: float = 0.0  # Typical power usage
    power_watts_max: float = 0.0  # Peak power usage

    capacity: float = 0.0
    ports: int = 0
    count: int = 1

    attrs: Dict[str, Any] = field(default_factory=dict)
    children: Dict[str, Component] = field(default_factory=dict)

    def total_cost(self) -> float:
        """Computes the total (recursive) cost of this component, including children,
        multiplied by this component's count.

        Returns:
            float: The total cost.
        """
        single_instance_cost = self.cost
        for child in self.children.values():
            single_instance_cost += child.total_cost()
        return single_instance_cost * self.count

    def total_power(self) -> float:
        """Computes the total *typical* (recursive) power usage of this component,
        including children, multiplied by this component's count.

        Returns:
            float: The total typical power in watts.
        """
        single_instance_power = self.power_watts
        for child in self.children.values():
            single_instance_power += child.total_power()
        return single_instance_power * self.count

    def total_power_max(self) -> float:
        """Computes the total *peak* (recursive) power usage of this component,
        including children, multiplied by this component's count.

        Returns:
            float: The total maximum (peak) power in watts.
        """
        single_instance_power_max = self.power_watts_max
        for child in self.children.values():
            single_instance_power_max += child.total_power_max()
        return single_instance_power_max * self.count

    def total_capacity(self) -> float:
        """Computes the total (recursive) capacity of this component,
        including children, multiplied by this component's count.

        Returns:
            float: The total capacity (dimensionless or user-defined units).
        """
        single_instance_capacity = self.capacity
        for child in self.children.values():
            single_instance_capacity += child.total_capacity()
        return single_instance_capacity * self.count

    def as_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """Returns a dictionary containing all properties of this component.

        Args:
            include_children (bool): If True, recursively includes children.

        Returns:
            Dict[str, Any]: Dictionary representation of this component.
        """
        data = {
            "name": self.name,
            "component_type": self.component_type,
            "description": self.description,
            "cost": self.cost,
            "power_watts": self.power_watts,
            "power_watts_max": self.power_watts_max,
            "capacity": self.capacity,
            "ports": self.ports,
            "count": self.count,
            "attrs": dict(self.attrs),  # shallow copy
        }
        if include_children:
            data["children"] = {
                child_name: child.as_dict(True)
                for child_name, child in self.children.items()
            }
        return data


@dataclass
class ComponentsLibrary:
    """Holds a collection of named Components. Each entry is a top-level "template"
    that can be referenced for cost/power/capacity lookups, possibly with nested children.

    Example (YAML-like):
        components:
          BigSwitch:
            component_type: chassis
            cost: 20000
            power_watts: 1750
            capacity: 25600
            children:
              PIM16Q-16x200G:
                component_type: linecard
                cost: 1000
                power_watts: 10
                ports: 16
                count: 8
          200G-FR4:
            component_type: optic
            cost: 2000
            power_watts: 6
            power_watts_max: 6.5
    """

    components: Dict[str, Component] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Component]:
        """Retrieves a Component by its name from the library.

        Args:
            name (str): Name of the component.

        Returns:
            Optional[Component]: The requested Component or None if not found.
        """
        return self.components.get(name)

    def merge(
        self, other: ComponentsLibrary, override: bool = True
    ) -> ComponentsLibrary:
        """Merges another ComponentsLibrary into this one. By default (override=True),
        duplicate components in `other` overwrite those in the current library.

        Args:
            other (ComponentsLibrary): Another library to merge into this one.
            override (bool): If True, components in `other` override existing ones.

        Returns:
            ComponentsLibrary: This instance, updated in place.
        """
        for comp_name, comp_obj in other.components.items():
            if override or comp_name not in self.components:
                self.components[comp_name] = comp_obj
        return self

    def clone(self) -> ComponentsLibrary:
        """Creates a deep copy of this ComponentsLibrary.

        Returns:
            ComponentsLibrary: A new, cloned library instance.
        """
        return ComponentsLibrary(components=deepcopy(self.components))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComponentsLibrary:
        """Constructs a ComponentsLibrary from a dictionary of raw component definitions.

        Args:
            data (Dict[str, Any]): Raw component definitions.

        Returns:
            ComponentsLibrary: A newly constructed library.
        """
        # Normalize dictionary keys to handle YAML boolean keys
        normalized_data = normalize_yaml_dict_keys(data)
        components_map: Dict[str, Component] = {}
        for comp_name, comp_def in normalized_data.items():
            components_map[comp_name] = cls._build_component(comp_name, comp_def)
        return ComponentsLibrary(components=components_map)

    @classmethod
    def _build_component(cls, name: str, definition_data: Dict[str, Any]) -> Component:
        """Recursively constructs a single Component from a dictionary definition.

        Args:
            name (str): Name of the component.
            definition_data (Dict[str, Any]): Dictionary data for the component definition.

        Returns:
            Component: The constructed Component instance.
        """
        comp_type = definition_data.get("component_type", "generic")
        cost = float(definition_data.get("cost", 0.0))
        power = float(definition_data.get("power_watts", 0.0))
        power_max = float(definition_data.get("power_watts_max", 0.0))
        capacity = float(definition_data.get("capacity", 0.0))
        ports = int(definition_data.get("ports", 0))
        count = int(definition_data.get("count", 1))

        child_definitions = definition_data.get("children", {})
        # Normalize child dictionary keys to handle YAML boolean keys
        normalized_children = normalize_yaml_dict_keys(child_definitions)
        children_map: Dict[str, Component] = {}
        for child_name, child_data in normalized_children.items():
            children_map[child_name] = cls._build_component(child_name, child_data)

        recognized_keys = {
            "component_type",
            "cost",
            "power_watts",
            "power_watts_max",
            "capacity",
            "ports",
            "children",
            "attrs",
            "count",
            "description",
        }
        attrs: Dict[str, Any] = dict(definition_data.get("attrs", {}))
        # Normalize attrs keys to handle YAML boolean keys
        attrs = normalize_yaml_dict_keys(attrs)
        leftover_keys = {
            k: v for k, v in definition_data.items() if k not in recognized_keys
        }
        # Normalize leftover keys too
        leftover_keys = normalize_yaml_dict_keys(leftover_keys)
        attrs.update(leftover_keys)

        return Component(
            name=name,
            component_type=comp_type,
            description=definition_data.get("description", ""),
            cost=cost,
            power_watts=power,
            power_watts_max=power_max,
            capacity=capacity,
            ports=ports,
            count=count,
            attrs=attrs,
            children=children_map,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> ComponentsLibrary:
        """Constructs a ComponentsLibrary from a YAML string. If the YAML contains
        a top-level 'components' key, that key is used; otherwise the entire
        top-level is treated as component definitions.

        Args:
            yaml_str (str): A YAML-formatted string of component definitions.

        Returns:
            ComponentsLibrary: A newly built components library.

        Raises:
            ValueError: If the top-level is not a dictionary or if the 'components'
                key is present but not a dictionary.
        """
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level must be a dict in Components YAML.")

        components_data = data.get("components") or data
        if not isinstance(components_data, dict):
            raise ValueError("'components' must be a dict if present.")

        return cls.from_dict(components_data)
