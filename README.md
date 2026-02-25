# agent-evaluation
A repo consolidates agent evaluation methods and experiments.

| 方向                  | 推荐项目 / 方法                               |
| ------------------- | --------------------------------------- |
| 最快做出可运行 repo        | **OpenJudge / Opik**                    |
| Agent 专用评估          | **AgentBench + MCPAgentBench**          |
| 自己评估方法实现            | **LLM-as-a-Judge 框架 + rubric 系统**       |
| 多模型对比评估             | **lm-evaluation-harness / opencompass** |
| 底层 observability 支撑 | Phoenix、Langfuse（非评估主体但可结合）             |

1. reference-free evaluation
2. Self-consistency
3. Synthetic Test Set + LLM (LLM-generated benchmark)
4. Property-based Testing

# 🚀 Hierarchical Agent Evaluation Framework

> Evaluating Multi-Tool LLM Agents Beyond Final Answers

Modern LLM agents are no longer single-step QA systems.  
They decompose user queries, route across tools (NL2SQL, RAG, APIs), and synthesize multi-stage reasoning into final responses.

However, most evaluation frameworks still focus only on:

- Final answer correctness  
- Static benchmark accuracy  

This repository proposes a different perspective:

> Multi-stage, tool-grounded, hierarchical evaluation for LLM agents.

---

## 🔍 Why This Project?

As multi-tool agents become production systems, failure no longer happens only at the final response layer.

It happens at:

- ❌ Incorrect task decomposition  
- ❌ Wrong tool routing (SQL vs RAG)  
- ❌ Invalid SQL generation  
- ❌ Retrieval hallucination  
- ❌ Unsupported synthesis  

Evaluating only the final answer hides systemic failures.

This project introduces:

> A structured evaluation framework that inspects each decision layer of an agent pipeline.

---

## 🏗 Target Agent Architecture

This framework is designed for agents that:

1. Decompose user queries into sub-queries  
2. Route each sub-query to:
   - NL2SQL engine
   - RAG retrieval system
3. Synthesize tool responses into final answers  

```
Example pipeline:

User Query
↓
Decomposition
↓
Routing (SQL / RAG)
↓
Tool Execution
↓
Final Answer
```

---

## 🧠 Core Idea: Hierarchical Evaluation

Instead of asking:

> "Is the final answer correct?"

We ask:

1. Was the query decomposition reasonable?
2. Was the routing decision appropriate?
3. Was the tool output correctly utilized?
4. Is the final answer faithful to tool outputs?

This enables deeper reliability analysis and better production debugging.

---

## 📊 Evaluation Dimensions

### 1️⃣ Decomposition Quality
- Coverage of user intent  
- Logical consistency  
- Redundancy detection  

### 2️⃣ Routing Accuracy
- Correct tool selection (SQL vs RAG)  
- LLM-based oracle comparison  
- Decision consistency  

### 3️⃣ Tool-Level Validation

#### SQL Evaluation
- Syntax validation  
- Execution success  
- Required column presence  
- Aggregation correctness  

#### RAG Evaluation
- Context grounding  
- Hallucination detection  
- Attribution verification  

### 4️⃣ Faithfulness Evaluation
- Final answer grounded in tool outputs  
- No unsupported claims  
- No contradictions  

### 5️⃣ Stability & Consistency
- Multi-run self-consistency  
- Embedding similarity analysis  
- Variance measurement  

---

## 🛠 Design Principles

- ✅ Fully local Python-based evaluation  
- ✅ No dependency on SaaS observability platforms  
- ✅ Extensible metric system  
- ✅ Reference-free evaluation supported  
- ✅ Suitable for CI/CD regression testing  

---

## 🏗 Repository Structure
```
agent-eval-framework/
│
├── evaluator/
│ ├── decomposition_eval.py
│ ├── routing_eval.py
│ ├── sql_eval.py
│ ├── rag_eval.py
│ ├── faithfulness_eval.py
│
├── core/
│ ├── runner.py
│ ├── metrics.py
│ ├── report.py
│
├── examples/
│ ├── sample_queries.json
│
└── README.md
```

---

## 🚀 Why This Matters

As LLM agents move into production environments, evaluation must shift from:

> Model-centric benchmarking  

to  

> System-level reliability engineering  

This project explores evaluation as:

- Agent observability  
- Decision validation  
- Tool-grounded reasoning verification  
- AI reliability engineering  

---

## 🎯 Vision

The long-term goal is to evolve this framework into:

- A reusable agent evaluation engine  
- A reliability scoring system  
- A regression testing toolkit for production agents  
- A foundation for trustworthy AI system design  

---

## 👤 Author

Built as part of a transition from GenAI Data Scientist to AI Systems Architect,  
with a focus on agent reliability, evaluation design, and AI product engineering.
```

