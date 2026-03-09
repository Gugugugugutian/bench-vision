from .BaseJudger import BaseJudger

class Llama3Judger(BaseJudger):
    def __init__(self):
        super().__init__()

    def _llama3_normalize(self, text):
        return self._normalize_output(text, split_key="assistant\n\n")

    def _judge_single(self, qid, question, answer, solution):
        return {
            "qid": qid,
            "question": question,
            "answer": answer,
            "solution": solution,
            "score": self._llama3_normalize(answer) == self._llama3_normalize(solution) 
        }, 1 if self._llama3_normalize(answer) == self._llama3_normalize(solution) else 0
    
    def judge(self, model_output_list):
        total_count = len(model_output_list)
        total_score = 0

        results = []
        for item in model_output_list:
            qid = item["qid"]
            question = item["question"]
            answer = item["answer"]
            solution = item["solution"]
            result, score = self._judge_single(qid, question, answer, solution)
            results.append(result)
            total_score += score
        return results, total_score / total_count if total_count > 0 else 0.0