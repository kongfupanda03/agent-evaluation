# agent-evaluation
A repo consolidates agent evaluation methods and experiments.

| 方向                  | 推荐项目 / 方法                               |
| ------------------- | --------------------------------------- |
| 最快做出可运行 repo        | **OpenJudge / Opik**                    |
| Agent 专用评估          | **AgentBench + MCPAgentBench**          |
| 自己评估方法实现            | **LLM-as-a-Judge 框架 + rubric 系统**       |
| 多模型对比评估             | **lm-evaluation-harness / opencompass** |
| 底层 observability 支撑 | Phoenix、Langfuse（非评估主体但可结合）             |

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

Example pipeline:

