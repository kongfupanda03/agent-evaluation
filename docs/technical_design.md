# 📐 Technical Design Document
## Hierarchical Agent Evaluation Framework

---

# 1. System Overview

## 1.1 Objective

Design a modular and extensible evaluation framework for multi-tool LLM agents that:

- Decompose user queries
- Route sub-queries to NL2SQL or RAG
- Execute tools
- Synthesize final answers

The framework evaluates each decision layer independently and aggregates results into system-level reliability metrics.

---

## 1.2 Design Goals

- Fully local Python-based execution
- No SaaS dependency
- Tool-grounded evaluation
- Reference-free evaluation supported
- CI/CD compatible
- Extensible to new tools

---

# 2. Target Agent Architecture

```
User Query
    ↓
Decomposition
    ↓
Routing Decision
    ↓
+-----------+------------+
|           |            |
|   SQL     |    RAG     |
|           |            |
+-----------+------------+
        ↓
    Tool Output
        ↓
    Final Answer
```

---

# 3. Evaluation Architecture

Evaluation mirrors the agent pipeline.

```
User Query
    ↓
Decomposition Evaluator
    ↓
Routing Evaluator
    ↓
Tool Evaluator
    ↓
Faithfulness Evaluator
    ↓
Stability Evaluator
    ↓
Metric Aggregator
    ↓
System Reliability Score
```

---

# 4. Evaluation Modules

---

## 4.1 Decomposition Evaluation

### Input

- Original user query
- Agent-generated sub-queries

### Method

LLM-as-a-Judge rubric scoring.

### Scoring Components

- Coverage score (0–1)
- Redundancy penalty (0–1)
- Logical consistency (0–1)

### Final Score Formula

```
DecompositionScore =
0.5 * Coverage
+ 0.3 * LogicalConsistency
+ 0.2 * (1 - Redundancy)
```

---

## 4.2 Routing Evaluation

### Objective

Validate tool selection correctness.

### Strategy

Use LLM Oracle:

```
Given the user query:
Which tool should be used? (SQL or RAG)
```

### Score

```
RoutingScore = 1 if AgentRoute == OracleRoute else 0
```

Optional extensions:

- Confidence-weighted routing
- Multi-tool routing scoring

---

## 4.3 SQL Evaluation Module

### Checks

- Syntax validity
- Execution success
- Non-empty result
- Required column presence
- Aggregation correctness

### Score Formula

```
SQLScore =
0.25 * Syntax
+ 0.25 * Execution
+ 0.25 * NonEmpty
+ 0.25 * SchemaMatch
```

---

## 4.4 RAG Evaluation Module

### Objectives

- Detect hallucination
- Ensure grounding
- Verify attribution

### Judge Prompt

```
Given:
Retrieved context
Final answer

Is the answer fully supported by context?
```

### Score Formula

```
RAGScore =
GroundingScore * (1 - HallucinationPenalty)
```

---

## 4.5 Faithfulness Evaluation

### Objective

Ensure final answer aligns strictly with tool outputs.

### Judge Questions

- Does the answer introduce unsupported claims?
- Does it contradict tool outputs?

### Score

```
FaithfulnessScore = 1 - UnsupportedClaimRate
```

---

## 4.6 Stability & Consistency

Run agent multiple times (N runs).

Compute:

- Pairwise cosine similarity
- Variance of answers

### Stability Score

```
StabilityScore =
Average pairwise embedding similarity
```

High variance indicates lower reliability.

---

# 5. Aggregation Layer

Combine all module scores:

```
SystemScore =
0.20 * DecompositionScore
+ 0.20 * RoutingScore
+ 0.25 * ToolScore
+ 0.25 * FaithfulnessScore
+ 0.10 * StabilityScore
```

Default weights can be customized depending on system requirements.

---

# 6. Execution Flow

```
for query in dataset:

    agent_output = run_agent(query)

    decomposition_score = eval_decomposition(...)
    routing_score = eval_routing(...)
    tool_score = eval_tool(...)
    faithfulness_score = eval_faithfulness(...)
    stability_score = eval_stability(...)

    aggregate_scores()
```

---

# 7. Extensibility Design

## 7.1 Base Evaluator Interface

```
class BaseEvaluator:
    def evaluate(self, input_data):
        raise NotImplementedError
```

All evaluation modules implement:

- evaluate()
- standardized metric dictionary output

---

## 7.2 Tool Plugin System

```
evaluator/
    sql_eval.py
    rag_eval.py
    api_eval.py
    graph_eval.py
```

Registry pattern:

```
EVALUATOR_REGISTRY = {
    "sql": SQLEvaluator(),
    "rag": RAGEvaluator(),
}
```

This allows dynamic expansion to new tool types.

---

## 7.3 Metric Registry

Support dynamic metric injection:

```
METRIC_REGISTRY = {
    "faithfulness": FaithfulnessEvaluator(),
    "stability": StabilityEvaluator(),
}
```

---

# 8. CI/CD Integration

Supports:

- Regression testing
- Model version comparison
- Tool version comparison
- Threshold-based deployment gating

Example:

```
if SystemScore < 0.75:
    raise DeploymentBlocker()
```

---

# 9. Future Extensions

- Drift detection
- Failure clustering
- Automatic adversarial test generation
- Error taxonomy classification
- Agent decision trace scoring
- Counterfactual robustness testing

---

# 10. Positioning

This framework is not a simple benchmark script.

It is:

- An Agent Reliability Engine
- A System-Level Evaluation Layer
- A Production AI Observability Foundation
- A Step Toward Trustworthy AI Systems
