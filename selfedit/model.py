from mlx_lm import batch_generate, generate, load
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3.5-0.8B-bf16"


class Model:
    def __init__(self, model_id: str = MODEL_ID, thinking: bool = False):
        self.model, self.tok = load(model_id)
        self.thinking = thinking

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        """Tokenize a chat history into the generation prompt's token ids."""
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking
        )
        return self.tok.encode(text, add_special_tokens=False)

    def sample_ids(self, prompt: list[int], n: int, temp: float, max_tokens: int = 1024) -> list[list[int]]:
        """Generate n continuations of a token prompt in one batch, returning their token ids."""
        sampler = make_sampler(temp=temp) if temp > 0 else None
        res = batch_generate(self.model, self.tok, [prompt] * n, max_tokens=max_tokens,
                             return_token_ids=True, sampler=sampler)
        return res.token_ids

    def decode(self, ids: list[int]) -> str:
        """Decode generated token ids to text."""
        return self.tok.decode(ids)

    def chat(self, messages: list[dict], temp: float = 0.0, max_tokens: int = 1024) -> str:
        """Generate a reply for a chat history at the given sampling temperature."""
        prompt = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking
        )
        return generate(
            self.model, self.tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp)
        )
