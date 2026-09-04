import json
import re
from typing import Any, Callable, Optional
from synapse.ai.providers import get_llm_provider, BaseLLMProvider


class ToolWrapper:
    def __init__(self, name: str, fn: Callable[..., Any], description: str = ""):
        self.name = name
        self.fn = fn
        self.description = description or f"Execute {name}"

    def __call__(self, *args, **kwargs) -> Any:
        return self.fn(*args, **kwargs)


class AgentRuntime:
    def __init__(
        self,
        name: str,
        model: str = "default",
        tools: Optional[list[Any]] = None,
        instructions: str = "",
        provider: Optional[BaseLLMProvider] = None,
        vm: Optional[Any] = None,
    ):
        self.name = name
        self.model = model
        self.tools: dict[str, Any] = {}
        self.vm = vm
        if tools:
            for t in tools:
                tool_name = getattr(t, "name", getattr(t, "__name__", None))
                if not tool_name and hasattr(t, "code"):
                    tool_name = getattr(t.code, "name", None)
                tool_name = tool_name or str(t)
                self.tools[tool_name] = t
        self.instructions = instructions
        self.provider = provider or get_llm_provider()
        self.memory: list[dict[str, str]] = []

    def __call__(self, task: Any) -> str:
        """Ajanın doğrudan fonksiyon veya boru hattı (|>) içinde çağrılabilmesini sağlar."""
        return self.run(str(task))

    def _execute_tool(self, tool_name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

        tool_obj = self.tools[tool_name]
        try:
            # SynapseFunction requires vm as the first argument
            if hasattr(tool_obj, "code") and hasattr(tool_obj, "env"):
                if self.vm is not None:
                    return tool_obj(self.vm, *args, **kwargs)
                return tool_obj(*args, **kwargs)
            elif callable(tool_obj):
                return tool_obj(*args, **kwargs)
            else:
                return f"Error: Tool '{tool_name}' is not callable"
        except Exception as e:
            return f"Tool execution error: {e}"

    def _parse_tool_call(self, response: str) -> Optional[tuple[str, list[Any], dict[str, Any]]]:
        """Model yanıtından tool çağrısını (JSON veya ReAct formatı) ayrıştırır."""
        cleaned = response.strip()

        # 1. Markdown JSON blokları
        if "```json" in cleaned:
            parts = cleaned.split("```json")
            if len(parts) > 1:
                json_candidate = parts[1].split("```")[0].strip()
                try:
                    data = json.loads(json_candidate)
                    if isinstance(data, dict) and ("tool" in data or "action" in data):
                        t_name = data.get("tool") or data.get("action")
                        args = data.get("args") or []
                        kwargs = data.get("kwargs") or data.get("parameters") or {}
                        if not isinstance(args, list):
                            args = [args]
                        return (str(t_name), args, kwargs)
                except Exception:
                    pass

        # 2. Doğrudan JSON arama ({...})
        match = re.search(r"\{[\s\S]*?\}", cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and ("tool" in data or "action" in data):
                    t_name = data.get("tool") or data.get("action")
                    args = data.get("args") or []
                    kwargs = data.get("kwargs") or data.get("parameters") or {}
                    if not isinstance(args, list):
                        args = [args]
                    return (str(t_name), args, kwargs)
            except Exception:
                pass

        # 3. Klasik ReAct kalıbı: Action: <tool> \n Action Input: <input>
        action_match = re.search(r"Action:\s*([A-Za-z0-9_]+)", cleaned)
        if action_match:
            t_name = action_match.group(1).strip()
            args = []
            kwargs = {}
            input_match = re.search(r"Action Input:\s*(.+)", cleaned)
            if input_match:
                inp_val = input_match.group(1).strip()
                try:
                    parsed_inp = json.loads(inp_val)
                    if isinstance(parsed_inp, list):
                        args = parsed_inp
                    elif isinstance(parsed_inp, dict):
                        kwargs = parsed_inp
                    else:
                        args = [parsed_inp]
                except Exception:
                    args = [inp_val]
            return (t_name, args, kwargs)

        return None

    def run(self, task: str, max_steps: int = 5) -> str:
        """Ajanın bir görevi otonom çalıştırma ReAct döngüsü."""
        if self.tools:
            tools_desc = []
            for t_name, t_obj in self.tools.items():
                doc = getattr(t_obj, "__doc__", "") or getattr(t_obj, "description", "")
                tools_desc.append(f"- {t_name}: {doc}")

            tools_info = "\n".join(tools_desc)
            system_prompt = (
                f"You are an AI agent named '{self.name}'. {self.instructions}\n"
                f"Available tools:\n{tools_info}\n\n"
                "To call a tool, reply ONLY with a JSON object:\n"
                '```json\n{"tool": "<name>", "args": [...]}\n```\n'
                "When the final answer is ready, output your answer directly."
            )
        else:
            system_prompt = f"You are an AI agent named '{self.name}'. {self.instructions}"

        self.memory.append({"role": "user", "content": task})
        current_input = f"Task: {task}"

        for step in range(max_steps):
            response = self.provider.generate(
                system_prompt=system_prompt,
                user_prompt=current_input
            )
            self.memory.append({"role": "assistant", "content": response})

            # Eğer alet yoksa veya tool çağrısı içermiyorsa doğrudan döndür
            if not self.tools:
                return response

            tool_call = self._parse_tool_call(response)
            if not tool_call:
                # Tool çağrısı yok, nihai cevap
                if "Final Answer:" in response:
                    return response.split("Final Answer:", 1)[1].strip()
                return response

            tool_name, args, kwargs = tool_call
            observation = self._execute_tool(tool_name, args, kwargs)

            obs_msg = f"Observation for {tool_name}: {observation}"
            self.memory.append({"role": "user", "content": obs_msg})
            current_input = f"{current_input}\nAssistant: {response}\n{obs_msg}"

        return response
