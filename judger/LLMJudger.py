from vllm import LLM, SamplingParams

from .BaseJudger import BaseJudger

class LLMJudger(BaseJudger):
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 800,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        )

    def _build_prompt(self, question, answer, solution):
        return (
            "You are a strict evaluator. Determine whether the model Answer and "
            "the Reference Solution are semantically equivalent for the given Question.\n\n"
            "Rules:\n"
            "- Return only YES or NO.\n"
            "- Ignore minor wording differences.\n"
            "- If the answer is incorrect, return NO.\n"
            "- If the answer is incomplete or ambiguous, return NO.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Reference Solution: {solution}\n\n"
            "Final: "
        )

    def _parse_yes_no(self, text):
        if text is None:
            return "no"
        t = text.split('assistantfinal')[-1].strip().lower()
        if t.startswith("yes"):
            return "yes"
        if t.startswith("no"):
            return "no"
        return "no"

    def _judge_single(self, qid, question, answer, solution):
        prompt = self._build_prompt(question, answer, solution)
        outputs = self.llm.generate([prompt], self.sampling_params)
        out_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        verdict = self._parse_yes_no(out_text)
        score = 1.0 if verdict == "yes" else 0.0
        return {
            "qid": qid,
            "question": question,
            "answer": answer,
            "solution": solution,
            "judge_output": out_text.strip(),
            "score": score,
        }, score

    def judge(self, model_output_list):
        total_count = len(model_output_list)
        total_score = 0.0
        results = []

        prompts = []
        for item in model_output_list:
            prompts.append(self._build_prompt(item["question"], item["answer"], item["solution"]))

        outputs = self.llm.generate(prompts, self.sampling_params)
        for item, out in zip(model_output_list, outputs):
            out_text = out.outputs[0].text if out.outputs else ""
            verdict = self._parse_yes_no(out_text)
            score = 1.0 if verdict == "yes" else 0.0
            results.append(
                {
                    "qid": item["qid"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "solution": item["solution"],
                    "judge_output": out_text.strip(),
                    "score": score,
                }
            )
            total_score += score

        return results, total_score / total_count if total_count > 0 else 0.0
