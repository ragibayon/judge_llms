import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MODEL_ID = "prometheus-eval/prometheus-7b-v2.0"
SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective "
    "feedback based on specific criteria, ensuring each assessment reflects "
    "absolute standards for performance."
)
PROMPT_NAMES = (
    "coherence",
    "factual_consistency",
    "fairness",
    "fluency",
    "relevance",
)
RESULT_RE = re.compile(
    r"Feedback:\s*(.*?)\s*\[RESULT\]\s*([1-5])",
    re.DOTALL,
)

env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=False,
    lstrip_blocks=False,
)

PROMPT_DICT = {
    name: env.get_template(f"{name}/prompts.j2")
    for name in PROMPT_NAMES
}


@dataclass
class EvaluationResult:
    domain: str
    score: int | None
    feedback: str
    raw_output: str
    prompt: str


def render_prompt(name: str, **context: str) -> str:
    return PROMPT_DICT[name].render(**context).strip()


def parse_feedback_and_score(text: str) -> tuple[str, int | None]:
    match = RESULT_RE.search(text)
    if not match:
        return text.strip(), None
    return match.group(1).strip(), int(match.group(2))


class PrometheusEvaluator:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device_map: str = "auto",
        torch_dtype: Any = None,
        max_new_tokens: int = 512,
    ) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError(
                "transformers and torch are required to run Prometheus evaluation. "
                "Install project dependencies first, for example with `uv sync`."
            )

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs = {
            "dtype": torch_dtype or self._default_dtype(),
        }
        resolved_device_map = self._resolve_device_map(device_map)
        if resolved_device_map is not None:
            load_kwargs["device_map"] = resolved_device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    def _default_dtype(self) -> Any:
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def _resolve_device_map(self, requested: str | None) -> str | None:
        if requested is None:
            return None
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "auto"
        return None

    def build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str) -> str:
        messages = self.build_messages(prompt)
        model_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(model_input, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    def evaluate_domain(
        self,
        domain: str,
        *,
        summary: str,
        safe_source: str,
        domain_task: str,
        domain_format: str = "summary",
    ) -> EvaluationResult:
        prompt = render_prompt(
            domain,
            domain_format=domain_format,
            domain_task=domain_task,
            safe_source=safe_source,
            summary=summary,
        )
        raw_output = self.generate(prompt)
        feedback, score = parse_feedback_and_score(raw_output)
        return EvaluationResult(
            domain=domain,
            score=score,
            feedback=feedback,
            raw_output=raw_output,
            prompt=prompt,
        )

    def evaluate_all(
        self,
        *,
        summary: str,
        safe_source: str,
        domain_task: str,
        domain_format: str = "summary",
    ) -> dict[str, EvaluationResult]:
        return {
            domain: self.evaluate_domain(
                domain,
                summary=summary,
                safe_source=safe_source,
                domain_task=domain_task,
                domain_format=domain_format,
            )
            for domain in PROMPT_NAMES
        }


if __name__ == "__main__":
    smoke_source = (
        "The city council approved a $2 million park renovation plan. "
        "Supporters said the upgrade would improve safety and accessibility, "
        "while critics argued the budget should prioritize road repairs first."
    )
    smoke_summary = (
        "The city council approved a $2 million renovation for the park. "
        "Backers said it would improve safety and accessibility, while critics "
        "said road repairs should come first."
    )
    smoke_task = "Summarize the article faithfully."

    evaluator = PrometheusEvaluator()
    results = evaluator.evaluate_all(
        summary=smoke_summary,
        safe_source=smoke_source,
        domain_task=smoke_task,
    )

    for name, result in results.items():
        print(f"=== {name} ===")
        print(f"score: {result.score}")
        print(f"feedback: {result.feedback}")
        print()
