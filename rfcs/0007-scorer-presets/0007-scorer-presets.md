---

## start_date: 2026-04-23

mlflow_issue: [https://github.com/mlflow/mlflow/issues/21445](https://github.com/mlflow/mlflow/issues/21445)
rfc_pr:

# Scorer Presets for Common Evaluation Patterns


| Author(s)              | Nehanth     |
| ---------------------- | ----------- |
| **Date Last Modified** | 2026-06-01  |
| **AI Assistant(s)**    | Claude Code |


# Summary

> **Note:** This RFC is based on [mlflow/mlflow#21445](https://github.com/mlflow/mlflow/issues/21445). The motivation, proposed presets, and API examples are derived from that issue, with additional design details and implementation specifics added here.

MLflow provides 21 built-in scorers for evaluating GenAI outputs, but users have no way to select a coherent subset for a specific evaluation pattern. Today, evaluating an agent requires importing and instantiating 9+ individual scorer classes -- boilerplate that gets copy-pasted across teams and templates.

This RFC proposes a `Preset` class that packages a named collection of scorers with support for **customization** and **persistence**. MLflow ships three built-in preset subclasses (`Rag`, `Agent`, `ConversationalAgent`) as starting points. Users can create custom presets, persist them to the MLflow server, and share them across teams and sessions. Presets can be passed directly in the `scorers` list alongside individual scorers, with automatic deduplication when presets overlap.

# Basic Example

```python
import mlflow
from mlflow.genai.scorers import Agent

# Use a built-in preset directly
result = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[Agent()],
)
```

```python
# Mix presets and individual scorers
from mlflow.genai.scorers import Agent, Guidelines

result = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[Agent(), Guidelines(name="tone", guidelines=["Respond professionally"])],
)
```

```python
# Combine presets using | -- duplicates are resolved automatically
from mlflow.genai.scorers import Agent, Rag

# Overlapping scorers (e.g. Safety, RelevanceToQuery) run once, not twice
scorers = Agent() | Rag()
result = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=scorers,
)
```

```python
# Define a custom preset and persist it for team sharing
from mlflow.genai.scorers import Preset, Safety, Fluency

my_preset = Preset("my_team_eval", scorers=[Safety(), Fluency(), my_custom_scorer])

# Register to MLflow server so the team can reuse it
my_preset.register()

# Later, another team member loads it
from mlflow.genai.scorers import get_preset

preset = get_preset(name="my_team_eval")
result = mlflow.genai.evaluate(data=eval_dataset, scorers=[preset])
```

## Motivation

### The Problem

As described in [the original issue](https://github.com/mlflow/mlflow/issues/21445), the Databricks agent app template [evaluate_agent.py](https://github.com/databricks/app-templates/blob/main/agent-openai-agents-sdk/agent_server/evaluate_agent.py) imports and instantiates 9 separate scorers to evaluate a conversational agent:

```python
from mlflow.genai.scorers import (
    Completeness,
    ConversationalSafety,
    ConversationCompleteness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    Safety,
    ToolCallCorrectness,
    UserFrustration,
)

mlflow.genai.evaluate(
    data=simulator,
    predict_fn=predict_fn,
    scorers=[
        Completeness(),
        ConversationCompleteness(),
        ConversationalSafety(),
        KnowledgeRetention(),
        UserFrustration(),
        Fluency(),
        RelevanceToQuery(),
        Safety(),
        ToolCallCorrectness(),
    ],
)
```

Every team building agent evaluation follows this same pattern. This creates three problems (from the [original issue](https://github.com/mlflow/mlflow/issues/21445)):

1. **No built-in grouping.** `get_all_scorers()` returns all 19 default-constructible scorers. Users evaluating a RAG pipeline get `ToolCallCorrectness`; users evaluating an agent get `RetrievalGroundedness`. Each unnecessary scorer wastes an LLM API call.
2. **21 scorers to choose from.** Users must read documentation for each scorer to determine relevance. Session-level scorers (e.g., `KnowledgeRetention`) silently produce no results when passed to single-turn evaluation.
3. **Copy-paste problem.** The same scorer lists get duplicated across templates, notebooks, and tutorials. When new scorers are added, existing lists don't pick them up.
4. **No persistence or sharing.** Teams cannot save and share a curated set of scorers. Each team member independently assembles their own list, leading to drift across projects.

### Who Benefits

- **New users** get a curated starting point without reading all 21 scorer docs
- **Teams** can define, persist, and share custom presets across sessions and team members
- **Template authors** replace hardcoded scorer lists with a single preset
- **MLflow maintainers** gain a single place to update when new scorers are added

### Out of Scope

- **Third-party scorer presets.** Integrating presets for DeepEval, RAGAS, or TruLens scorers.

## Detailed Design

### The `Preset` Class

A `Preset` is a named, iterable container of scorers. It is **not** a `Scorer` subclass -- it is a grouping mechanism that gets flattened into individual scorers at validation time.

```python
class Preset:
    """A named, immutable collection of scorers for a common evaluation pattern.

    Presets can be passed in the ``scorers`` list alongside individual
    scorers. They are flattened during validation, so the evaluation
    loop only ever sees individual ``Scorer`` instances.

    Args:
        name: A descriptive name for this preset.
        scorers: The list of scorer instances in this preset.
    """

    def __init__(self, name: str, scorers: list[Scorer]):
        self._name = name
        self._validate_no_duplicates(scorers)
        self._scorers = tuple(scorers)

    @staticmethod
    def _validate_no_duplicates(scorers):
        seen = set()
        for scorer in scorers:
            key = (type(scorer), scorer.name)
            if key in seen:
                raise MlflowException.invalid_parameter_value(
                    f"Duplicate scorer: {type(scorer).__name__} with name '{scorer.name}'. "
                    "Use different names for scorers of the same type."
                )
            seen.add(key)

    @staticmethod
    def _deduplicate(scorers):
        seen = set()
        result = []
        for scorer in scorers:
            key = (type(scorer), scorer.name)
            if key not in seen:
                seen.add(key)
                result.append(scorer)
        return result

    @property
    def name(self) -> str:
        return self._name

    @property
    def scorers(self) -> tuple:
        return self._scorers

    def __iter__(self):
        return iter(self._scorers)

    def __len__(self):
        return len(self._scorers)

    def __or__(self, other):
        if isinstance(other, (Preset, list)):
            combined = list(self) + list(other)
            return self._deduplicate(combined)
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, list):
            combined = other + list(self)
            return self._deduplicate(combined)
        return NotImplemented

    def register(self, *, experiment_id: str | None = None):
        """Register this preset to the MLflow server for team sharing."""
        ...

    def __repr__(self):
        scorer_names = [type(s).__name__ for s in self._scorers]
        return f"Preset('{self._name}', [{', '.join(scorer_names)}])"
```

**Key design decisions:**

- **Immutable.** Scorers are stored as a tuple and exposed via a read-only property.
- **Blocks duplicates on construction.** `__init__` raises an error if duplicate scorers (same type and name) are passed. This is explicit — users know immediately if they have a conflict, rather than duplicates being silently removed.
- **Set union via `|`.** Supports `__or__`/`__ror__` for combining presets with deduplication: `Agent() | [my_scorer]` or `Agent() | Rag()`. Uses `|` instead of `+` because the deduplication behavior matches set union semantics. Deduplication on `|` is silent because combining presets with overlapping scorers is expected usage.
- **Not a `Scorer` subclass.** A preset doesn't produce feedback -- it's a container. The evaluation loop assumes one scorer = one result column. Making `Preset` a scorer would require changes throughout the pipeline (aggregation, telemetry, serialization).
- **Stores instances, not classes.** Users pass already-configured scorer instances.

### Built-in Presets as Subclasses

Each built-in preset is a subclass of `Preset` that hardcodes its scorer list. This means each call creates **fresh scorer instances** (no shared mutable singletons) and supports preset-specific customization.

```python
class Agent(Preset):
    def __init__(self):
        super().__init__("agent", [
            ToolCallCorrectness(),
            ToolCallEfficiency(),
            RelevanceToQuery(),
            Safety(),
            Completeness(),
        ])

class Rag(Preset):
    def __init__(self):
        super().__init__("rag", [
            RetrievalRelevance(),
            RetrievalGroundedness(),
            RelevanceToQuery(),
            Safety(),
            Completeness(),
        ])

class ConversationalAgent(Preset):
    def __init__(self):
        super().__init__("conversational-agent", [
            ToolCallCorrectness(),
            ToolCallEfficiency(),
            RelevanceToQuery(),
            Safety(),
            Completeness(),
            UserFrustration(),
            ConversationCompleteness(),
            ConversationalSafety(),
            ConversationalToolCallEfficiency(),
            KnowledgeRetention(),
        ])
```

**Why subclasses over instances:**

- **Fresh instances every time.** `Agent()` creates new scorer instances on each call. No shared mutable state.
- **Preset-specific customization.** Each preset can accept its own parameters (e.g., `Agent(model="openai:/gpt-4o")` to set the judge model for all scorers).
- **Type checking.** `isinstance(preset, Agent)` works — code can distinguish which preset is being used.
- **Custom control flow.** Each preset can override methods for preset-specific validation or behavior.

### Customization

Users can customize presets in several ways:

**Combine with additional scorers using `|`:**

```python
scorers = Agent() | [Fluency(), Guidelines(name="tone", guidelines=["Be professional"])]
```

**Create a custom preset from scratch:**

```python
my_preset = Preset("my_eval", scorers=[
    ToolCallCorrectness(),
    Safety(),
    my_custom_scorer,
])
```

**Subclass a built-in preset to add defaults:**

```python
class MyAgent(Agent):
    def __init__(self):
        super().__init__()
        # Add team-specific scorers
        self._scorers = self._scorers + (Fluency(), my_compliance_scorer)
```

### Persistence

Presets can be registered to the MLflow server so teams can share them across sessions. This leverages the existing scorer registration infrastructure.

**Register a preset:**

```python
my_preset = Preset("my_team_agent", scorers=[
    ToolCallCorrectness(),
    Safety(),
    Fluency(),
])

# Register to the active experiment
my_preset.register()

# Or register to a specific experiment
my_preset.register(experiment_id="123")
```

**Load a persisted preset:**

```python
from mlflow.genai.scorers import get_preset

# Load from the active experiment
preset = get_preset(name="my_team_agent")

# Load from a specific experiment
preset = get_preset(name="my_team_agent", experiment_id="123")

result = mlflow.genai.evaluate(data=eval_dataset, scorers=[preset])
```

**Why persistence matters:**

- **Version stability.** Persisted presets are snapshots — they don't change when MLflow upgrades. Built-in presets serve as starting points; teams persist their own versions for stability.
- **Team sharing.** A persisted preset is available to any team member with access to the experiment.
- **Customization without code.** Teams can customize and persist presets without modifying source code or templates.

### Deduplication

When presets are combined using `|`, the same scorer type can appear more than once. For example, `Agent()` and `Rag()` both contain `Safety()` and `RelevanceToQuery()`. Running the same scorer twice wastes LLM API calls and produces duplicate result columns.

Deduplication happens in two places:

- **In `__or__`** — when presets are combined using `|`, duplicates are removed using `(type(scorer), scorer.name)` as the key. This is expected behavior when combining presets with overlapping scorers.
- **In `validate_scorers()`** — when multiple presets are passed directly in a list (e.g., `scorers=[Agent(), Rag()]`) without using `|`, `__or__` is never called. `validate_scorers()` flattens and deduplicates as a safety net:

```python
def validate_scorers(scorers: list[Any]) -> list[Scorer]:
    from mlflow.genai.scorers.presets import Preset

    # 1. Flatten presets into individual scorers
    flat = []
    for item in scorers:
        if isinstance(item, Preset):
            flat.extend(item)
        else:
            flat.append(item)

    # 2. Deduplicate by (type, name)
    flat = Preset._deduplicate(flat)

    # 3. Existing validation on the flattened list
    ...
```

Scorers of the same class with different names are preserved (e.g., two `Guidelines` with different rules). Only true duplicates — same class and same name — are removed.

`evaluate()` itself does not change. By the time scorers reach the evaluation loop, they are all individual `Scorer` instances.

### Built-in Preset Summary

MLflow ships three built-in preset subclasses as starting points. Each call creates fresh scorer instances. Users can customize and persist their own presets.

| Preset                 | Scorers                                                                                                                                 | Use Case                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `Rag()`                | RetrievalRelevance, RetrievalGroundedness, RelevanceToQuery, Safety, Completeness                                                       | Retrieval-augmented generation pipelines                 |
| `Agent()`              | ToolCallCorrectness, ToolCallEfficiency, RelevanceToQuery, Safety, Completeness                                                         | Single-turn tool-calling agents                          |
| `ConversationalAgent()`| All of `Agent` + UserFrustration, ConversationCompleteness, ConversationalSafety, ConversationalToolCallEfficiency, KnowledgeRetention  | Multi-turn conversational agents                         |


#### Design Rationale

- **Safety is in `Rag` and `Agent`** because these presets aim to be complete starting points. Most users want safety checks without composing two presets.
- **Fluency is excluded from `Agent`** because agent evaluation emphasizes tool usage and task completion. Users who need it can compose: `Agent() | [Fluency()]`.
- **`ConversationalAgent` excludes `ConversationalRoleAdherence`** because it requires a defined persona in the system prompt, which not all agents have.
- **`RetrievalSufficiency` is excluded from `Rag`** because it requires `expected_response` or `expected_facts` (ground truth). Users who have expectations data can add it manually: `Rag() | [RetrievalSufficiency()]`.
- **`Correctness` is excluded from all presets** because it requires `expectations` (ground truth) data.
- **`Guidelines` and `ConversationalGuidelines` are excluded from all presets** because both require a `guidelines` constructor argument.
- **Only three built-in presets** (Rag, Agent, ConversationalAgent) — these represent clear, distinct evaluation patterns. Other groupings (e.g., safety, quality) are too vague or too small to justify a built-in preset. Users can create and persist their own groupings for their specific needs.

## Drawbacks

1. **New class in the API.** Adds `Preset` to the public surface. Mitigation: it's a simple container with persistence support.
2. **Opinionated defaults.** Not everyone will agree on which scorers belong in which preset. Mitigation: presets are extensible via `|`, and users can create and persist their own.
3. **Persistence adds scope.** Supporting preset registration and retrieval increases implementation complexity. Mitigation: leverages the existing scorer registration infrastructure.

# Alternatives

### 1. `get_preset()` function (no class)

Instead of a `Preset` class, provide a simple function that returns a plain list. This approach is simpler and also supports persistence and customization.

Usage:

```python
from mlflow.genai.scorers import get_preset

# Simple usage
result = mlflow.genai.evaluate(scorers=get_preset("agent"))

# Extending a preset
scorers = get_preset("agent") + [Guidelines(name="tone", guidelines=["Be professional"])]
result = mlflow.genai.evaluate(scorers=scorers)

# Create a custom preset and persist it
register_preset("my_team_agent", scorers=[Safety(), Fluency(), my_custom_scorer])

# Load it later
scorers = get_preset("my_team_agent")
result = mlflow.genai.evaluate(scorers=scorers)
```

**Pros:** Simpler. No validation changes needed. Returns fresh instances each call. `Literal` type gives IDE autocompletion. Going from function to class later is non-breaking. Can also support persistence by registering and loading presets by name.

**Cons:** No user-defined preset objects. Composition requires `+` with list concatenation. The preset concept disappears immediately -- it's just a list. No deduplication when combining presets. If this approach is preferred, the RFC can be updated to use it. This is a viable alternative if the class approach is deemed too heavy.

### 2. Tag-based filtering

Add `categories` to each scorer class and provide `get_scorers(categories=["rag"])`. More flexible but over-engineered for 21 scorers and requires modifying every existing class.

### 3. Enum-based API

`ScorerPreset.RAG.get_scorers()`. Type-safe but heavier API surface.

### 4. Do nothing

Users keep copy-pasting scorer lists. Does not scale as the scorer count grows.

# Adoption Strategy

This is an **additive, non-breaking change**. Existing code continues to work unchanged.

- Update documentation and templates to show `Preset` usage alongside the manual import pattern.
- Update the `validate_scorers()` error message to mention presets for discoverability.
- Databricks agent templates can simplify from 9 imports + 9 instantiations to `scorers=[ConversationalAgent()]`.
- Teams can persist their customized presets and share them across projects.

# Open Questions

1. **Class-based vs function-based approach.** The class-based approach is proposed as the primary design for its ergonomics and customization support. The function-based approach is a viable alternative that may be more flexible for persistence. Both approaches were discussed during review.
