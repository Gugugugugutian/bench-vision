class BaseJudger:
    def __init__(
            self, 
            judge_method: str = "str_match",
            question_key: str = "question",
            answer_key: str = "answer",
            solution_key: str = "solution", 
            debug: bool = False
        ):
        self.judge_method = judge_method
        self.question_key = question_key
        self.answer_key = answer_key
        self.solution_key = solution_key
        self.debug = debug

    def _debug(self, message):
        if self.debug:
            print(f"[Judger] {message}")

    def _normalize_output(self, output, split_key = None):
        if split_key is not None:
            normalized = output.split(split_key)[-1].strip().rstrip('.').lower()
        normalized = output.strip().rstrip('.').lower()
        self._debug(f"Normalized output: '{normalized}'")
        return normalized

    def _judge_single(self, qid, question, answer, solution):
        raise NotImplementedError("Subclasses must implement this method")

    def judge(self, model_output_list):
        raise NotImplementedError("Subclasses must implement this method")