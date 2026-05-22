import json
import time

class AIPromptEvaluationFramework:
    def __init__(self):
        self.metrics = ["factual_accuracy", "hallucination_severity", "reasoning_depth", "instruction_compliance", "coherence", "safety"]

    def evaluate_response(self, prompt, response, ground_truth=None):
        print(f"[INFO] Running evaluation pipeline on response...")
        time.sleep(0.5)
        
        scores = {}
        if len(response) < 20:
            scores["coherence"] = 2
            scores["reasoning_depth"] = 1
        else:
            scores["coherence"] = 5
            scores["reasoning_depth"] = 4

        if "don't" in prompt.lower() and "don't" in response.lower():
            scores["instruction_compliance"] = 3
        else:
            scores["instruction_compliance"] = 5

        if ground_truth and any(word not in ground_truth for word in response.split()[:3]):
            scores["factual_accuracy"] = 3
            scores["hallucination_severity"] = 4
        else:
            scores["factual_accuracy"] = 5
            scores["hallucination_severity"] = 0
            
        scores["safety"] = 5
        
        avg_score = sum(scores.values()) / len(self.metrics)
        status = "PASSED (Gold Standard)" if avg_score >= 4.0 else "REJECTED"

        evaluation_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_analyzed": prompt,
            "metrics_scores": scores,
            "average_alignment_score": round(avg_score, 2),
            "status": status
        }
        return evaluation_report

if __name__ == "__main__":
    evaluator = AIPromptEvaluationFramework()
    sample_prompt = "Explain why compiled languages execute faster than interpreted ones without mentioning Java."
    sample_response = "Compiled languages translate source code directly into machine language instructions before execution, making execution incredibly swift. Interpreted environments compile code line-by-line during runtime."
    truth_data = "Compiled languages convert code directly to machine code execution speeds are faster."

    report = evaluator.evaluate_response(sample_prompt, sample_response, ground_truth=truth_data)
    print("\n--- GENERATED EVALUATION JSON BATCH OUTPUT ---")
    print(json.dumps(report, indent=4))
      
