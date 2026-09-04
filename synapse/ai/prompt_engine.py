import json
from typing import Any, Optional
from synapse.ai.providers import get_llm_provider, BaseLLMProvider


class PromptEngine:
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or get_llm_provider()

    def execute(
        self,
        system_template: str,
        user_template: str,
        params: dict[str, Any],
        temperature: float = 0.7,
        schema: Optional[dict[str, Any]] = None
    ) -> Any:
        # String template formatlama
        system_text = self._format_template(system_template, params)
        user_text = self._format_template(user_template, params)

        raw_result = self.provider.generate(
            system_prompt=system_text,
            user_prompt=user_text,
            temperature=temperature,
            json_schema=schema
        )

        if schema:
            try:
                # Markdown ```json blokları varsa ayıkla
                cleaned = raw_result.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                return json.loads(cleaned)
            except Exception:
                return raw_result

        return raw_result

    def _format_template(self, template: str, params: dict[str, Any]) -> str:
        if template in params:
            return str(params[template])
        res = str(template)
        for k, v in params.items():
            res = res.replace(f"{{{k}}}", str(v))
        return res
