from .Llama3Judger import Llama3Judger

try:
    from .LLMJudger import LLMJudger
except Exception:
    LLMJudger = None
