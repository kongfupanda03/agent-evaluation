# agent-evaluation
A repo consolidates agent evaluation methods and experiments.

| 方向                  | 推荐项目 / 方法                               |
| ------------------- | --------------------------------------- |
| 最快做出可运行 repo        | **OpenJudge / Opik**                    |
| Agent 专用评估          | **AgentBench + MCPAgentBench**          |
| 自己评估方法实现            | **LLM-as-a-Judge 框架 + rubric 系统**       |
| 多模型对比评估             | **lm-evaluation-harness / opencompass** |
| 底层 observability 支撑 | Phoenix、Langfuse（非评估主体但可结合）             |



🔥 Agent Reliability Evaluation Framework

包括：

Relevance Score (LLM Judge)

Hallucination Detection

Tool-grounded Verification

Self-consistency Score

Format Validation

Stability Benchmark

输出：

每个问题的评分

汇总报告

JSON log

可视化图表

agent-eval-framework/
│
├── evaluator/
│   ├── routing_eval.py
│   ├── decomposition_eval.py
│   ├── sql_eval.py
│   ├── rag_eval.py
│   ├── faithfulness_eval.py
│
├── core/
│   ├── runner.py
│   ├── metrics.py
│   ├── report.py
│
├── examples/
│   ├── sample_queries.json
│
└── README.md
