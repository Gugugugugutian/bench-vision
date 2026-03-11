import re

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
        if output is None:
            return ""
        text = str(output)
        if split_key is not None:
            text = text.split(split_key)[-1]
        text = text.strip()
        text = text.strip('"').strip("'").strip()
        text = text.rstrip(" \t\n\r.。!！?？,:;；")
        normalized = text.lower()
        self._debug(f"Normalized output: '{normalized}'")
        return normalized

    def _normalize_for_match(self, text):
        if text is None:
            return ""
        text = str(text).lower()
        text = text.replace("&", " and ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text):
        norm = self._normalize_for_match(text)
        return [t for t in norm.split(" ") if t]

    def _normalize_number_str(self, s):
        if "." in s:
            int_part, frac = s.split(".", 1)
            int_part = int_part.lstrip("0") or "0"
            frac = frac.rstrip("0")
            return int_part if frac == "" else f"{int_part}.{frac}"
        return s.lstrip("0") or "0"

    def _extract_number_values(self, text):
        if text is None:
            return []
        text = str(text).lower()
        values = []
        for m in re.finditer(r"\d+(?:\.\d+)?", text):
            s = m.group(0)
            # Guard against pathological long digit sequences
            if len(re.sub(r"\\D", "", s)) > 12:
                continue
            values.append(self._normalize_number_str(s))
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        }
        tokens = self._tokenize(text)
        for t in tokens:
            if t in word_to_num:
                values.append(str(word_to_num[t]))
        return values

    def _extract_option_letters(self, text):
        if text is None:
            return []
        text = str(text).upper()
        letters = re.findall(r"\b[A-J]\b", text)
        return letters

    def _is_yes_no(self, text):
        t = self._normalize_for_match(text)
        return t in {"yes", "no"}

    def _match_yes_no(self, answer, solution):
        sol = self._normalize_for_match(solution)
        ans = self._normalize_for_match(answer)
        if sol not in {"yes", "no"}:
            return False
        if sol == "yes":
            return ans.startswith("yes") or ans.startswith("y")
        return ans.startswith("no") or ans.startswith("n")

    def _match_numbers(self, answer, solution, question=None):
        sol_vals = self._extract_number_values(solution)
        ans_vals = self._extract_number_values(answer)
        if not sol_vals:
            return False
        if len(sol_vals) == 1 and len(ans_vals) == 1 and ans_vals[0] == sol_vals[0]:
            return True
        if question is not None:
            q = self._normalize_for_match(question)
            if any(k in q for k in ["how many", "number", "numbers", "year", "years", "date", "time"]):
                return all(v in ans_vals for v in sol_vals)
        return False

    def _is_ocr_question(self, question):
        if question is None:
            return False
        q = self._normalize_for_match(question)
        keywords = [
            "text", "spell", "say", "says", "saying", "quote", "word", "words",
            "written", "title", "name", "sign", "label", "brand", "logo",
            "number", "numbers", "date", "year", "years", "price",
            "address", "phone", "website", "url", "email"
        ]
        return any(k in q for k in keywords)

    def _remove_stopwords(self, tokens):
        stop = {
            "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "for",
            "is", "are", "was", "were", "be", "by", "with", "from", "this",
            "that", "these", "those", "it", "its", "as", "di"
        }
        return [t for t in tokens if t not in stop]

    def _edit_distance_leq1(self, a, b):
        if a == b:
            return True
        if abs(len(a) - len(b)) > 1:
            return False
        if len(a) > len(b):
            a, b = b, a
        # now len(b) >= len(a)
        i = j = diffs = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
                j += 1
            else:
                diffs += 1
                if diffs > 1:
                    return False
                if len(a) == len(b):
                    i += 1
                    j += 1
                else:
                    j += 1
        return True

    def _match_option_letters(self, answer, solution):
        sol_letters = self._extract_option_letters(solution)
        ans_letters = self._extract_option_letters(answer)
        if not sol_letters or not ans_letters:
            return False
        return sorted(set(sol_letters)) == sorted(set(ans_letters))

    def _match_contains(self, answer, solution, question=None):
        ans_norm = self._normalize_for_match(answer)
        sol_norm = self._normalize_for_match(solution)
        if not sol_norm:
            return False
        ans_tokens = ans_norm.split(" ") if ans_norm else []
        sol_tokens = sol_norm.split(" ") if sol_norm else []
        if len(sol_tokens) == 1:
            if sol_tokens[0] in ans_tokens:
                return True
            if self._is_ocr_question(question) and len(ans_tokens) == 1:
                return self._edit_distance_leq1(ans_tokens[0], sol_tokens[0])
            return False
        if len(sol_tokens) <= 3:
            return f" {sol_norm} " in f" {ans_norm} "
        if len(sol_tokens) <= 6:
            # allow subsequence match for short phrases (handles inserted fillers like "volume")
            it = iter(ans_tokens)
            if all(any(t == a for a in it) for t in sol_tokens):
                return True
        if self._is_ocr_question(question):
            ans_ns = self._remove_stopwords(ans_tokens)
            sol_ns = self._remove_stopwords(sol_tokens)
            if sol_ns and all(t in ans_ns for t in sol_ns):
                return True
            if ans_ns and sol_ns:
                shorter, longer = (ans_ns, sol_ns) if len(ans_ns) <= len(sol_ns) else (sol_ns, ans_ns)
                if all(t in longer for t in shorter):
                    overlap = len(shorter) / max(len(longer), 1)
                    if overlap >= 0.8:
                        return True
        # URL or identifier-like answers: try raw substring match
        raw_sol = self._normalize_output(solution)
        raw_ans = self._normalize_output(answer)
        if any(ch in raw_sol for ch in [".", "/", "-"]):
            return raw_sol in raw_ans
        return False

    def _judge_single(self, qid, question, answer, solution):
        raise NotImplementedError("Subclasses must implement this method")

    def judge(self, model_output_list):
        raise NotImplementedError("Subclasses must implement this method")
