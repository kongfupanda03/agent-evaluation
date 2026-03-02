"""
Faithfulness Evaluator
Evaluates whether the final answer is faithful to the tool outputs
Uses DeepEval for faithfulness scoring
"""

import os

# Disable DeepEval telemetry BEFORE any imports
os.environ["CONFIDENT_AI_API_KEY"] = ""
os.environ["DEEPEVAL_TELEMETRY"] = "OFF"
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["DEEPEVAL_ALLOW_TELEMETRY"] = "NO"
os.environ["POSTHOG_API_KEY"] = ""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()





@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation"""
    score: float = 0.0
    passed: bool = False
    reason: str = ""
    metric_name: str = ""


class FaithfulnessEvaluator:
    """
    Evaluates faithfulness of agent's final answer to tool outputs
    
    Based on technical_design.md Section 4.5:
    - Ensures final answer aligns strictly with tool outputs
    - Detects unsupported claims
    - Checks for contradictions with tool outputs
    """
    
    def __init__(self, threshold: float = 0.7, model: str = "gpt-4o-mini"):
        """
        Initialize the faithfulness evaluator
        
        Args:
            threshold: Minimum score to pass (0-1)
            model: LLM model to use for evaluation
        """
        self.threshold = threshold
        self.model = model
        self._setup_metric()
    
    def _setup_metric(self):
        """Setup DeepEval faithfulness metric"""
        # Create custom LLM wrapper for DeepEval
        class LangChainLLM(DeepEvalBaseLLM):
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.model = ChatOpenAI(
                    model=model_name,
                    temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            
            def load_model(self):
                return self.model
            
            def generate(self, prompt: str) -> str:
                return self.model.invoke(prompt).content
            
            async def a_generate(self, prompt: str) -> str:
                return self.model.invoke(prompt).content
            
            def get_model_name(self) -> str:
                return self.model_name
        
        # Initialize the metric with custom LLM
        custom_llm = LangChainLLM(self.model)
        
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=custom_llm,
            include_reason=True
        )
        
        # Initialize GEval metric for MCQ correctness
        # Note: GEval uses model name string, not custom LLM wrapper
        self.correctness_metric = GEval(
            name="MCQ Correctness",
            criteria="""Evaluate whether the actual_output correctly matches the expected_output for multiple choice questions (MCQ).

EVALUATION RULES (apply in order of priority):

1. OPTION SELECTION MATCH (Highest Priority):
   - First, extract the selected option(s) from both actual_output and expected_output
   - Options can be: letters (A, B, C, D), numbers (1, 2, 3), scores (score 1, score 2), or descriptive labels
   - If the selected OPTION differs (e.g., actual="A" vs expected="B", or actual="score 1" vs expected="score 2"), the answer is INCORRECT regardless of semantic similarity
   - For multi-select questions, ALL selected options must match exactly

2. SEMANTIC EQUIVALENCE (Only if option selection matches or for open-ended questions):
   - If options match or no clear options exist, check if the meaning and content are semantically equivalent
   - Consider paraphrasing, different wording with same meaning as acceptable

INCORRECT EXAMPLES:
- Expected: "B. company provides financial estimates for a single scenario", Actual: "A. company provides financial estimates for a range of scenarios" → INCORRECT (different option)
- Expected: "score 2", Actual: "score 1&2" → INCORRECT (different selection)
- Expected: "C", Actual: "A" → INCORRECT (different letter option)

CORRECT EXAMPLES:
- Expected: "B", Actual: "The answer is B" → CORRECT (same option B)
- Expected: "score 3", Actual: "score 3" → CORRECT (same option)
- Expected: "A and C", Actual: "Options A, C" → CORRECT (same multi-select)""",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=self.model  # Use model name string
        )
    
    def evaluate(
        self,
        query: str,
        tool_outputs: Dict[str, Any],
        final_answer: str
    ) -> FaithfulnessResult:
        """
        Evaluate faithfulness of final answer to tool outputs
        
        Args:
            query: Original user query
            tool_outputs: Dictionary of tool outputs from agent execution
            final_answer: Final synthesized answer from agent
            
        Returns:
            FaithfulnessResult with score and assessment
        """
        # Combine tool outputs into retrieval context
        retrieval_context = self._format_tool_outputs(tool_outputs)
        
        # Create test case for DeepEval
        test_case = LLMTestCase(
            input=query,
            actual_output=final_answer,
            retrieval_context=[retrieval_context]
        )
        
        # Run faithfulness evaluation
        self.faithfulness_metric.measure(test_case)
        
        return FaithfulnessResult(
            score=self.faithfulness_metric.score,
            passed=self.faithfulness_metric.is_successful(),
            reason=self.faithfulness_metric.reason,
            metric_name="faithfulness"
        )
    
    def evaluate_correctness(
        self,
        actual_output: str,
        expected_output: str,
        query: str = ""
    ) -> FaithfulnessResult:
        """
        Evaluate MCQ correctness using GEval
        Checks if actual answer matches expected answer
        
        Args:
            actual_output: The actual answer selected (e.g., "A", "B", "The answer is C")
            expected_output: The expected correct answer (e.g., "A", "B", "C")
            query: Optional query context
            
        Returns:
            FaithfulnessResult with correctness score
        """
        test_case = LLMTestCase(
            input=query or "MCQ evaluation",
            actual_output=actual_output,
            expected_output=expected_output
        )
        
        self.correctness_metric.measure(test_case)
        
        return FaithfulnessResult(
            score=self.correctness_metric.score,
            passed=self.correctness_metric.is_successful(),
            reason=self.correctness_metric.reason,
            metric_name="correctness"
        )
    
    def evaluate_query_answered(
        self,
        query: str,
        final_answer: str
    ) -> FaithfulnessResult:
        """
        Custom evaluation: Check if final answer actually answers the user query
        Uses LLM-as-judge to verify answer addresses query intent
        
        Args:
            query: Original user query
            final_answer: Final synthesized answer from agent
            
        Returns:
            FaithfulnessResult with score and assessment
        """
        llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        judge_prompt = f"""You are an evaluator assessing whether an answer properly addresses a user query.

User Query: {query}

Final Answer: {final_answer}

Evaluate whether the final answer:
1. Directly addresses what the user asked for
2. Provides the information requested (even if in structured/tabular format)
3. Does not ignore parts of the query

Score from 0.0 to 1.0:
- 1.0: Perfectly answers the query
- 0.7-0.9: Good answer, minor gaps
- 0.4-0.6: Partial answer, missing some information
- 0.0-0.3: Does not answer the query or completely wrong

Respond in this exact format:
SCORE: [number between 0.0 and 1.0]
REASON: [brief explanation of why this score was given]
"""
        
        response = llm.invoke(judge_prompt).content
        
        # Parse score and reason
        import re
        score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', response)
        reason_match = re.search(r'REASON:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        
        score = float(score_match.group(1)) if score_match else 0.0
        reason = reason_match.group(1).strip() if reason_match else "Could not parse evaluation"
        
        return FaithfulnessResult(
            score=score,
            passed=score >= self.threshold,
            reason=reason,
            metric_name="query_answered"
        )
    
    def evaluate_comprehensive(
        self,
        query: str,
        tool_outputs: Dict[str, Any],
        final_answer: str
    ) -> Dict[str, FaithfulnessResult]:
        """
        Comprehensive evaluation:
        - Faithfulness: Checks answer is grounded in tool outputs (no hallucination)
        - Query Answered: Checks answer actually addresses user query
        
        Args:
            query: Original user query
            tool_outputs: Dictionary of tool outputs from agent execution
            final_answer: Final synthesized answer from agent
            
        Returns:
            Dictionary with both evaluation results
        """
        # Run faithfulness evaluation (for RAG grounding)
        faithfulness_result = self.evaluate(query, tool_outputs, final_answer)
        
        # Run query-answered evaluation
        query_answered_result = self.evaluate_query_answered(query, final_answer)
        
        return {
            "faithfulness": faithfulness_result,
            "query_answered": query_answered_result
        }
    
    def _format_tool_outputs(self, tool_outputs: Dict[str, Any]) -> str:
        """
        Format tool outputs into a string for retrieval context
        
        Args:
            tool_outputs: Dictionary of tool outputs
            
        Returns:
            Formatted string of tool outputs
        """
        formatted_parts = []
        
        for tool_name, outputs in tool_outputs.items():
            formatted_parts.append(f"=== {tool_name.upper()} OUTPUT ===")
            
            if isinstance(outputs, list):
                for i, output in enumerate(outputs, 1):
                    if isinstance(output, dict):
                        query = output.get("query", "")
                        result = output.get("result", "")
                        formatted_parts.append(f"[{i}] Query: {query}")
                        formatted_parts.append(f"    Result: {result}")
                    else:
                        formatted_parts.append(f"[{i}] {output}")
            elif isinstance(outputs, dict):
                for key, value in outputs.items():
                    formatted_parts.append(f"{key}: {value}")
            else:
                formatted_parts.append(str(outputs))
            
            formatted_parts.append("")  # Empty line between tools
        
        return "\n".join(formatted_parts)
    
    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[FaithfulnessResult]:
        """
        Evaluate multiple test cases in batch
        
        Args:
            test_cases: List of dicts with keys: query, tool_outputs, final_answer
            
        Returns:
            List of FaithfulnessResult
        """
        results = []
        
        for case in test_cases:
            result = self.evaluate(
                query=case["query"],
                tool_outputs=case["tool_outputs"],
                final_answer=case["final_answer"]
            )
            results.append(result)
        
        return results


def main():
    """
    Example usage of FaithfulnessEvaluator
    """
    # Example test case
    query = "What is ABC Corp's current loan exposure and ESG commitments?"
    
    tool_outputs = {
        "nl2sql_tool": [
            {
                "query": "What is ABC Corp's loan exposure?",
                "result": "ABC Corp has total loan exposure of $50M across Loans and Trade Finance products."
            }
        ],
        "rag_tool": [
            {
                "query": "Find ABC Corp's ESG commitments",
                "result": "ABC Corp has committed to net-zero carbon emissions by 2040, with 25% reduction already achieved since 2020."
            }
        ]
    }
    
    # Faithful answer (should score high)
    faithful_answer = (
        "ABC Corp has a current loan exposure of $50M across Loans and Trade Finance products. "
        "In terms of ESG commitments, ABC Corp has pledged to achieve net-zero carbon emissions by 2040 "
        "and has already reduced emissions by 25% since 2020."
    )
    
    # Unfaithful answer (should score low - introduces unsupported claim)
    unfaithful_answer = (
        "ABC Corp has a loan exposure of $100M and plans to achieve carbon neutrality by 2030. "
        "They have also committed to 100% renewable energy."
    )
    
    # Initialize evaluator
    evaluator = FaithfulnessEvaluator(threshold=0.7)
    
    print("=" * 60)
    print("Faithfulness Evaluation Example")
    print("=" * 60)
    
    # Evaluate faithful answer
    print("\n--- Faithful Answer ---")
    result1 = evaluator.evaluate(query, tool_outputs, faithful_answer)
    print(f"Score: {result1.score:.2f}")
    print(f"Passed: {result1.passed}")
    print(f"Reason: {result1.reason}")
    
    # Evaluate unfaithful answer
    print("\n--- Unfaithful Answer ---")
    result2 = evaluator.evaluate(query, tool_outputs, unfaithful_answer)
    print(f"Score: {result2.score:.2f}")
    print(f"Passed: {result2.passed}")
    print(f"Reason: {result2.reason}")
    
    # Evaluate with comprehensive metrics
    print("\n--- Comprehensive Evaluation (Faithful) ---")
    combined = evaluator.evaluate_comprehensive(query, tool_outputs, faithful_answer)
    for metric_name, result in combined.items():
        print(f"\n{metric_name}:")
        print(f"  Score: {result.score:.2f}")
        print(f"  Passed: {result.passed}")
        print(f"  Reason: {result.reason}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    main()
