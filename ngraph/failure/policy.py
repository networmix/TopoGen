"""Failure policy primitives.

Defines `FailureCondition`, `FailureRule`, and `FailurePolicy` for expressing
how nodes, links, and risk groups fail in analyses. Conditions match on
top-level attributes with simple operators; rules select matches using
"all", probabilistic "random" (with `probability`), or fixed-size "choice"
(with `count`). Policies can optionally expand failures by shared risk groups
or by risk-group children.
"""

from __future__ import annotations

import random as _random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set


@dataclass
class FailureCondition:
    """A single condition for matching an entity's attribute with an operator and value.

    Example usage (YAML):
      conditions:
        - attr: "capacity"
          operator: "<"
          value: 100

    Attributes:
        attr (str):
            The name of the attribute to inspect (e.g., "capacity", "region").
        operator (str):
            The comparison operator: "==", "!=", "<", "<=", ">", ">=",
            "contains", "not_contains", "any_value", or "no_value".
        value (Any):
            The value to compare against (e.g., 100, True, "foo", etc.).
    """

    attr: str
    operator: str
    value: Any


# Supported entity scopes for a rule
EntityScope = Literal["node", "link", "risk_group"]


@dataclass
class FailureRule:
    """Defines how to match and then select entities for failure.

    Attributes:
        entity_scope (EntityScope):
            The type of entities this rule applies to: "node", "link", or "risk_group".
        conditions (List[FailureCondition]):
            A list of conditions to filter matching entities.
        logic (Literal["and", "or"]):
            "and": All conditions must be true for a match.
            "or": At least one condition is true for a match (default).
        rule_type (Literal["random", "choice", "all"]):
            The selection strategy among the matched set:
              - "random": each matched entity is chosen with probability = `probability`.
              - "choice": pick exactly `count` items from the matched set (random sample).
              - "all": select every matched entity in the matched set.
        probability (float):
            Probability in [0,1], used if `rule_type="random"`.
        count (int):
            Number of entities to pick if `rule_type="choice"`.
    """

    entity_scope: EntityScope
    conditions: List[FailureCondition] = field(default_factory=list)
    logic: Literal["and", "or"] = "or"
    rule_type: Literal["random", "choice", "all"] = "all"
    probability: float = 1.0
    count: int = 1

    def __post_init__(self) -> None:
        if self.rule_type == "random":
            if not (0.0 <= self.probability <= 1.0):
                raise ValueError(
                    f"probability={self.probability} must be within [0,1] "
                    f"for rule_type='random'."
                )


@dataclass
class FailurePolicy:
    """A container for multiple FailureRules plus optional metadata in `attrs`.

    The main entry point is `apply_failures`, which:
      1) For each rule, gather the relevant entities (node, link, or risk_group).
              2) Match them based on rule conditions using 'and' or 'or' logic.
      3) Apply the selection strategy (all, random, or choice).
      4) Collect the union of all failed entities across all rules.
      5) Optionally expand failures by shared-risk groups or sub-risks.

    Large-scale performance:
      - If you set `use_cache=True`, matched sets for each rule are cached,
        so repeated calls to `apply_failures` can skip re-matching if the
        network hasn't changed. If your network changes between calls,
        you should clear the cache or re-initialize the policy.

    Example YAML configuration:
        ```yaml
        failure_policy:
          attrs:
            description: "Regional power grid failure affecting telecom infrastructure"
          fail_risk_groups: true
          rules:
            # Fail all nodes in Texas electrical grid
            - entity_scope: "node"
              conditions:
                - attr: "electric_grid"
                  operator: "=="
                  value: "texas"
              logic: "and"
              rule_type: "all"

            # Randomly fail 40% of underground fiber links in affected region
            - entity_scope: "link"
              conditions:
                - attr: "region"
                  operator: "=="
                  value: "southwest"
                - attr: "installation"
                  operator: "=="
                  value: "underground"
              logic: "and"
              rule_type: "random"
              probability: 0.4

            # Choose exactly 2 risk groups to fail (e.g., data centers)
            # Note: logic defaults to "or" when not specified
            - entity_scope: "risk_group"
              rule_type: "choice"
              count: 2
        ```

    Attributes:
        rules (List[FailureRule]):
            A list of FailureRules to apply.
        attrs (Dict[str, Any]):
            Arbitrary metadata about this policy (e.g. "name", "description").
        fail_risk_groups (bool):
            If True, after initial selection, expand failures among any
            node/link that shares a risk group with a failed entity.
        fail_risk_group_children (bool):
            If True, and if a risk_group is marked as failed, expand to
            children risk_groups recursively.
        use_cache (bool):
            If True, match results for each rule are cached to speed up
            repeated calls. If the network changes, the cached results
            may be stale.
        seed (Optional[int]):
            Seed for reproducible random operations. If None, operations
            will be non-deterministic.

    """

    rules: List[FailureRule] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    fail_risk_groups: bool = False
    fail_risk_group_children: bool = False
    use_cache: bool = False
    seed: Optional[int] = None

    # Internal cache for matched sets:  (rule_index -> set_of_entities)
    _match_cache: Dict[int, Set[str]] = field(default_factory=dict, init=False)

    def apply_failures(
        self,
        network_nodes: Dict[str, Any],
        network_links: Dict[str, Any],
        network_risk_groups: Dict[str, Any] | None = None,
        *,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Identify which entities fail given the defined rules, then optionally
        expand by shared-risk groups or nested risk groups.

        Args:
            network_nodes: {node_id -> node_object_or_dict}, each with top-level attributes
                           (capacity, disabled, risk_groups, etc.).
            network_links: {link_id -> link_object_or_dict}, similarly.
            network_risk_groups: {rg_name -> RiskGroup} or dict. If you don't have risk
                                 groups, pass None or {}.
            seed: Optional seed for deterministic selection. Overrides the policy's
                internal seed when provided.

        Returns:
            A list of IDs that fail (union of all rule matches, possibly expanded).
            For risk groups, the ID is the risk group's name.
        """
        if network_risk_groups is None:
            network_risk_groups = {}

        failed_nodes: Set[str] = set()
        failed_links: Set[str] = set()
        failed_risk_groups: Set[str] = set()

        # 1) Collect matched from each rule
        for idx, rule in enumerate(self.rules):
            matched_ids = self._match_scope(
                idx,
                rule,
                network_nodes,
                network_links,
                network_risk_groups,
            )
            # Then select a subset from matched_ids according to rule_type
            # When an explicit seed is provided, it overrides the policy's seed
            # for deterministic selection without creating a policy copy.
            effective_seed = seed if seed is not None else self.seed
            selected = self._select_entities(matched_ids, rule, effective_seed)

            # Add them to the respective fail sets
            if rule.entity_scope == "node":
                failed_nodes |= set(selected)
            elif rule.entity_scope == "link":
                failed_links |= set(selected)
            elif rule.entity_scope == "risk_group":
                failed_risk_groups |= set(selected)

        # 2) Optionally expand failures by risk groups
        if self.fail_risk_groups:
            self._expand_risk_groups(
                failed_nodes, failed_links, network_nodes, network_links
            )

        # 3) Optionally expand risk-group children (if a risk group is failed, recursively fail children)
        if self.fail_risk_group_children and failed_risk_groups:
            self._expand_failed_risk_group_children(
                failed_risk_groups, network_risk_groups
            )

        # Return union: node IDs, link IDs, and risk_group names
        # For the code that uses this, you can interpret them in your manager.
        all_failed = set(failed_nodes) | set(failed_links) | set(failed_risk_groups)
        return sorted(all_failed)

    def _match_scope(
        self,
        rule_idx: int,
        rule: FailureRule,
        network_nodes: Dict[str, Any],
        network_links: Dict[str, Any],
        network_risk_groups: Dict[str, Any],
    ) -> Set[str]:
        """Get the set of IDs matched by the given rule, either from cache
        or by performing a fresh match over the relevant entity type.
        """
        if self.use_cache and rule_idx in self._match_cache:
            return self._match_cache[rule_idx]

        # Decide which mapping to iterate
        if rule.entity_scope == "node":
            matched = self._match_entities(network_nodes, rule.conditions, rule.logic)
        elif rule.entity_scope == "link":
            matched = self._match_entities(network_links, rule.conditions, rule.logic)
        else:  # risk_group
            matched = self._match_entities(
                network_risk_groups, rule.conditions, rule.logic
            )

        if self.use_cache:
            self._match_cache[rule_idx] = matched
        return matched

    def _match_entities(
        self,
        entity_map: Dict[str, Any],
        conditions: List[FailureCondition],
        logic: str,
    ) -> Set[str]:
        """Return all entity IDs that match the given conditions based on 'and'/'or' logic.

        entity_map is either nodes, links, or risk_groups:
          {entity_id -> {top_level_attr: value, ...}}

        If no conditions, return everything (no restrictions means all match).

        Returns:
            A set of matching entity IDs.
        """
        if not conditions:
            # No conditions means match all entities regardless of logic
            return set(entity_map.keys())

        matched = set()
        for entity_id, attrs in entity_map.items():
            if self._evaluate_conditions(attrs, conditions, logic):
                matched.add(entity_id)

        return matched

    @staticmethod
    def _evaluate_conditions(
        attrs: Dict[str, Any],
        conditions: List[FailureCondition],
        logic: str,
    ) -> bool:
        """Evaluate multiple conditions on a single entity. All or any condition(s)
        must pass, depending on 'logic'.
        """
        if logic == "and":
            return all(_evaluate_condition(attrs, c) for c in conditions)
        elif logic == "or":
            return any(_evaluate_condition(attrs, c) for c in conditions)
        else:
            raise ValueError(f"Unsupported logic: {logic}")

    @staticmethod
    def _select_entities(
        entity_ids: Set[str], rule: FailureRule, seed: Optional[int] = None
    ) -> Set[str]:
        """From the matched IDs, pick which entities fail under the given rule_type.

        Args:
            entity_ids: Set of entity IDs that matched the rule conditions.
            rule: The FailureRule specifying how to select from matched entities.
            seed: Optional seed for reproducible random operations.

        Returns:
            Set of entity IDs selected for failure.
        """
        if not entity_ids:
            return set()

        if rule.rule_type == "random":
            if seed is not None:
                # Use seeded random state for deterministic results
                rng = _random.Random(seed)
                return {eid for eid in entity_ids if rng.random() < rule.probability}
            else:
                # Use global random state when no seed provided
                return {
                    eid for eid in entity_ids if _random.random() < rule.probability
                }
        elif rule.rule_type == "choice":
            count = min(rule.count, len(entity_ids))
            entity_list = list(entity_ids)
            if seed is not None:
                # Use seeded random state for deterministic results
                rng = _random.Random(seed)
                return set(rng.sample(entity_list, k=count))
            else:
                # Use global random state when no seed provided
                return set(_random.sample(entity_list, k=count))
        elif rule.rule_type == "all":
            return entity_ids
        else:
            raise ValueError(f"Unsupported rule_type: {rule.rule_type}")

    def _expand_risk_groups(
        self,
        failed_nodes: Set[str],
        failed_links: Set[str],
        network_nodes: Dict[str, Any],
        network_links: Dict[str, Any],
    ) -> None:
        """Expand failures among any node/link that shares a risk group
        with a failed entity. BFS until no new failures.
        """
        # We'll handle node + link expansions only. (Risk group expansions are separate.)
        # Build a map risk_group -> set of node or link IDs
        rg_to_entities: Dict[str, Set[str]] = defaultdict(set)

        # Gather risk_groups from nodes
        for n_id, nd in network_nodes.items():
            if "risk_groups" in nd and nd["risk_groups"]:
                for rg in nd["risk_groups"]:
                    rg_to_entities[rg].add(n_id)

        # Gather risk_groups from links
        for l_id, lk in network_links.items():
            if "risk_groups" in lk and lk["risk_groups"]:
                for rg in lk["risk_groups"]:
                    rg_to_entities[rg].add(l_id)

        # Combined set of failed node/link IDs
        queue = deque(failed_nodes | failed_links)
        visited = set(queue)  # track which entity IDs we've processed

        while queue:
            current_id = queue.popleft()
            # figure out if current_id is a node or a link by seeing where it appears
            current_rgs = []
            if current_id in network_nodes:
                # node
                nd = network_nodes[current_id]
                current_rgs = nd.get("risk_groups", [])
            elif current_id in network_links:
                # link
                lk = network_links[current_id]
                current_rgs = lk.get("risk_groups", [])

            for rg in current_rgs:
                # all entity IDs in rg_to_entities[rg] should be failed
                for other_id in rg_to_entities[rg]:
                    if other_id not in visited:
                        visited.add(other_id)
                        queue.append(other_id)
                        if other_id in network_nodes:
                            failed_nodes.add(other_id)
                        elif other_id in network_links:
                            failed_links.add(other_id)
                        # if other_id in risk_groups => not handled here

    def _expand_failed_risk_group_children(
        self,
        failed_rgs: Set[str],
        all_risk_groups: Dict[str, Any],
    ) -> None:
        """If we fail a risk_group, also fail its descendants recursively.

        We assume each entry in all_risk_groups is something like:
            rg_name -> RiskGroup object or { 'name': .., 'children': [...] }

        BFS or DFS any children to mark them as failed as well.
        """
        queue = deque(failed_rgs)
        while queue:
            rg_name = queue.popleft()
            rg_data = all_risk_groups.get(rg_name)
            if not rg_data:
                continue
            # Suppose the children are in rg_data["children"]
            # or if it's an actual RiskGroup object => rg_data.children
            child_list = []
            if isinstance(rg_data, dict):
                child_list = rg_data.get("children", [])
            else:
                # assume it's a RiskGroup object with a .children
                child_list = rg_data.children

            for child_obj in child_list:
                # child_obj might be a dict or RiskGroup with name
                child_name = (
                    child_obj["name"] if isinstance(child_obj, dict) else child_obj.name
                )
                if child_name not in failed_rgs:
                    failed_rgs.add(child_name)
                    queue.append(child_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields as JSON-serializable primitives.
        """
        return {
            "rules": [
                {
                    "entity_scope": rule.entity_scope,
                    "conditions": [
                        {
                            "attr": cond.attr,
                            "operator": cond.operator,
                            "value": cond.value,
                        }
                        for cond in rule.conditions
                    ],
                    "logic": rule.logic,
                    "rule_type": rule.rule_type,
                    "probability": rule.probability,
                    "count": rule.count,
                }
                for rule in self.rules
            ],
            "attrs": self.attrs,
            "fail_risk_groups": self.fail_risk_groups,
            "fail_risk_group_children": self.fail_risk_group_children,
            "use_cache": self.use_cache,
            "seed": self.seed,
        }


def _evaluate_condition(entity_attrs: Dict[str, Any], cond: FailureCondition) -> bool:
    """Evaluate a single FailureCondition against entity attributes.

    Operators supported:
      ==, !=, <, <=, >, >=
      contains, not_contains
      any_value, no_value

    If entity_attrs does not have cond.attr => derived_value=None.

    Returns True if condition passes, else False.
    """
    has_attr = cond.attr in entity_attrs
    derived_value = entity_attrs.get(cond.attr, None)
    op = cond.operator

    if op == "==":
        return derived_value == cond.value
    elif op == "!=":
        return derived_value != cond.value
    elif op == "<":
        return (derived_value is not None) and (derived_value < cond.value)
    elif op == "<=":
        return (derived_value is not None) and (derived_value <= cond.value)
    elif op == ">":
        return (derived_value is not None) and (derived_value > cond.value)
    elif op == ">=":
        return (derived_value is not None) and (derived_value >= cond.value)
    elif op == "contains":
        if derived_value is None:
            return False
        return cond.value in derived_value
    elif op == "not_contains":
        if derived_value is None:
            return True
        return cond.value not in derived_value
    elif op == "any_value":
        return has_attr  # True if attribute key exists
    elif op == "no_value":
        # Pass if the attribute key is missing or the value is None
        return (not has_attr) or (derived_value is None)
    else:
        raise ValueError(f"Unsupported operator: {op}")
