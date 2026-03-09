class BaseModel:
    def __init__(
            self, 
            model_path,
            checkpoint_path: str | None = None,
            temperature: float = 0.0,
            top_p: float = 1.0,
            max_tokens: int = 2048,
            top_k: int = 0,
            load_when_init: bool = True,
            debug: bool = False,
        ):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path

        self.model = None
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.debug = debug

        if load_when_init:
            self._debug(f"Loading model from {model_path}...")
            self.load_model()

    def _debug(self, message):
        if self.debug:
            print(f"[Model] {message}")

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(
            self, 
            input_data: list[dict], 
            id_key: str = "qid",
            prompt_key: str = "question",
            response_key: str = "answer", 
            image_key: str = "image", 
            other_keys_to_keep: list = ['solution', ], 
        ) -> list:
        raise NotImplementedError("Subclasses must implement this method")