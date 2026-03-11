from .BaseJudger import BaseJudger

class Llama3Judger(BaseJudger):
    def __init__(self):
        super().__init__()

    def _compute_score(self, question, answer, solution):
        ans = self._llama3_normalize(answer)
        sol = self._normalize_output(solution)

        if self._match_yes_no(ans, sol):
            return 1.0
        if self._match_numbers(ans, sol, question=question):
            return 1.0
        if self._match_option_letters(ans, sol):
            return 1.0
        if self._normalize_for_match(ans) == self._normalize_for_match(sol):
            return 1.0
        if self._match_contains(ans, sol, question=question):
            return 1.0
        return 0.0

    def _llama3_normalize(self, text):
        return self._normalize_output(text, split_key="assistant\n\n")

    def _judge_single(self, qid, question, answer, solution):
        score = self._compute_score(question, answer, solution)
        return {
            "qid": qid,
            "question": question,
            "answer": answer,
            "solution": solution,
            "score": score
        }, score
    
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
