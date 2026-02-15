"""
Faithfulness Evaluator
Evaluates whether the final answer is faithful to the tool outputs
Uses DeepEval for faithfulness scoring
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Disable DeepEval telemetry
os.environ["CONFIDENT_AI_API_KEY"] = ""
os.environ["DEEPEVAL_TELEMETRY"] = "OFF"
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
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
        
        self.relevancy_metric = AnswerRelevancyMetric(
            threshold=self.threshold,
            model=custom_llm,
            include_reason=True
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
    
    def evaluate_with_relevancy(
        self,
        query: str,
        tool_outputs: Dict[str, Any],
        final_answer: str
    ) -> Dict[str, FaithfulnessResult]:
        """
        Evaluate both faithfulness and answer relevancy
        
        Args:
            query: Original user query
            tool_outputs: Dictionary of tool outputs from agent execution
            final_answer: Final synthesized answer from agent
            
        Returns:
            Dictionary with both evaluation results
        """
        # Combine tool outputs into retrieval context
        retrieval_context = self._format_tool_outputs(tool_outputs)
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=final_answer,
            retrieval_context=[retrieval_context]
        )
        
        # Run both metrics
        self.faithfulness_metric.measure(test_case)
        self.relevancy_metric.measure(test_case)
        
        return {
            "faithfulness": FaithfulnessResult(
                score=self.faithfulness_metric.score,
                passed=self.faithfulness_metric.is_successful(),
                reason=self.faithfulness_metric.reason,
                metric_name="faithfulness"
            ),
            "answer_relevancy": FaithfulnessResult(
                score=self.relevancy_metric.score,
                passed=self.relevancy_metric.is_successful(),
                reason=self.relevancy_metric.reason,
                metric_name="answer_relevancy"
            )
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
    
    # Evaluate with both metrics
    print("\n--- Combined Metrics (Faithful) ---")
    combined = evaluator.evaluate_with_relevancy(query, tool_outputs, faithful_answer)
    for metric_name, result in combined.items():
        print(f"\n{metric_name}:")
        print(f"  Score: {result.score:.2f}")
        print(f"  Passed: {result.passed}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    main()
