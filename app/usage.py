from .config import Settings


MODEL_RATES = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "deepseek-v4-pro": (1.32, 3.96),
    "deepseek-v4-flash": (0.44, 1.32),
}


def rates(provider: str, settings: Settings, model: str | None = None) -> tuple[float, float]:
    if model in MODEL_RATES:
        return MODEL_RATES[model]
    if provider == "openai":
        return settings.openai_input_cost_per_million, settings.openai_output_cost_per_million
    return settings.deepseek_input_cost_per_million, settings.deepseek_output_cost_per_million


def calculate_cost(provider: str, input_tokens: int, output_tokens: int, settings: Settings, model: str | None = None) -> float:
    input_rate, output_rate = rates(provider, settings, model)
    return round((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 8)


def approximate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
