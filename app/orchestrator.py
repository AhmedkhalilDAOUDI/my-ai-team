import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from .config import Settings
from .providers import DeepSeekProvider, OpenAICompatibleProvider, OpenAIProvider, Provider, SYSTEM_PROMPT


@dataclass
class ProviderResult:
    status: str
    text: str = ""
    error: str = ""
    usage_provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class ThesisTeam:
    roles = {
        "openai": """ROLE: SUPERVISOR AND TECHNICAL BUILDER.
Own the solution. Make decisions, design the implementation, and turn ambiguity into a concrete plan or artifact. Focus on what should be built and how. Integrate valid criticism, but do not behave like a literature-search assistant or produce a generic pros/cons list.""",
        "deepseek": """ROLE: ADVERSARIAL CRITIC AND REVIEWER.
Do not independently recreate the builder's answer. Stress-test the proposal, identify the most consequential flaws and hidden assumptions, and give precise corrections. State a clear verdict. Prioritize correctness, feasibility, evaluation quality, and what could invalidate the work.""",
        "jury": """ROLE: JURY, powered by GPT-5.6 Sol.
Interrogate the work with the few decisive questions that expose weak reasoning. Judge competing positions against the user's goal, correctness, evidence, feasibility, and originality. Do not redo the builder's work. Give a clear verdict, required corrections, and the next question the team must answer.""",
        "auditor": """ROLE: COMPLETENESS AUDITOR, powered by DeepSeek V4 Pro.
Inspect the full work for consequential omissions, ignored constraints, unresolved dependencies, missing evidence, and blind spots. Do not repeat the supervisor, critic, or jury. Report only what is materially missing and why it matters.""",
    }
    labels = {"openai": "ChatGPT", "deepseek": "DeepSeek", "jury": "Jury", "auditor": "Completeness Auditor"}
    def __init__(self, settings: Settings):
        self.settings = settings
        timeout = settings.request_timeout_seconds
        self.providers: dict[str, Provider] = {
            "openai": OpenAIProvider(settings.openai_api_key, settings.openai_model, timeout, settings.max_output_tokens),
            "deepseek": DeepSeekProvider(settings.deepseek_api_key, settings.deepseek_model, timeout, settings.max_output_tokens),
            "jury": OpenAIProvider(settings.openai_api_key, settings.jury_model, timeout, settings.max_output_tokens),
            "auditor": DeepSeekProvider(settings.deepseek_api_key, settings.auditor_model, timeout, settings.max_output_tokens),
        }
        self.providers["jury"].name = "Jury / OpenAI GPT-5.6 Sol"
        self.providers["auditor"].name = "Completeness Auditor / DeepSeek V4 Pro"
        self.synthesis_provider = settings.synthesis_provider

    def _provider_for_agent(self, agent: dict) -> Provider:
        if agent["provider"] == "openai":
            provider = OpenAIProvider(
                self.settings.openai_api_key, agent["model"], self.settings.request_timeout_seconds,
                self.settings.max_output_tokens,
            )
        elif agent["provider"] == "deepseek":
            provider = DeepSeekProvider(
                self.settings.deepseek_api_key, agent["model"], self.settings.request_timeout_seconds,
                self.settings.max_output_tokens,
            )
        else:
            import json, os
            from .platform import PlatformStore
            from .storage import Store
            plugin = PlatformStore(Store()).plugin(agent["provider"])
            if not plugin or not plugin["enabled"] or plugin["adapter"] != "openai-compatible": raise ValueError(f"Unsupported or disabled provider: {agent['provider']}")
            provider = OpenAICompatibleProvider(plugin["name"], os.getenv(plugin["api_key_env"]), agent["model"], self.settings.request_timeout_seconds, self.settings.max_output_tokens, plugin["base_url"])
            provider.provider_id = plugin["name"]
        provider.name = f"{agent['agent_name']} / {agent['provider']}"
        return provider

    @staticmethod
    def _agent_system_prompt(agent: dict) -> str:
        limit = agent.get("max_sentences", 5)
        instructions = agent.get("instructions", "").strip()
        return f"""You are {agent['agent_name']}, a member of My AI Team.

ROLE:
{agent['role']}

SPECIFIC INSTRUCTIONS:
{instructions or '(none)'}

ANSWER CONTRACT:
- Lead with the answer; no greeting or restatement.
- Write no more than {limit} complete, information-dense sentences.
- Stay inside your assigned role and do not imitate other agents.
- Prefer specific decisions, corrections, evidence, and next actions.
- Finish every sentence; never cut the answer mid-sentence."""

    def status(self) -> dict[str, bool]:
        return {name: provider.configured for name, provider in self.providers.items()}

    async def _safe_ask(self, provider: Provider, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> ProviderResult:
        try:
            text = await provider.ask(prompt, system_prompt)
            return ProviderResult(status="ok", text=text, usage_provider=provider.provider_id, model=provider.model, **provider.last_usage)
        except Exception as exc:
            return ProviderResult(status="error", error=str(exc), usage_provider=getattr(provider, "provider_id", ""), model=provider.model)

    async def ask(self, prompt: str, route: str) -> tuple[dict[str, ProviderResult], ProviderResult]:
        results = {name: ProviderResult(status="skipped") for name in self.providers}
        results["openai"] = await self._safe_ask(self.providers["openai"], prompt, self._role_prompt("openai"))
        critic_prompt = f"User task:\n{prompt}\n\nSUPERVISOR ANSWER:\n{results['openai'].text}\n\nCritique this answer directly; do not replace it with an independent answer."
        results["deepseek"] = await self._safe_ask(self.providers["deepseek"], critic_prompt, self._role_prompt("deepseek"))
        jury_prompt = f"User task:\n{prompt}\n\nSUPERVISOR ANSWER:\n{results['openai'].text}\n\nCRITIC REVIEW:\n{results['deepseek'].text}\n\nAsk only 2-5 important questions needed to test or clarify these answers. Do not answer, judge, recommend, or summarize."
        results["jury"] = await self._safe_ask(self.providers["jury"], jury_prompt, self._role_prompt("jury"))
        auditor_prompt = f"User task:\n{prompt}\n\nSUPERVISOR ANSWER:\n{results['openai'].text}\n\nCRITIC REVIEW:\n{results['deepseek'].text}\n\nJURY QUESTIONS:\n{results['jury'].text}\n\nIdentify only important things still missing. Do not repeat earlier points, answer the jury, or rewrite the solution."
        results["auditor"] = await self._safe_ask(self.providers["auditor"], auditor_prompt, self._role_prompt("auditor"))
        successful = {name: item.text for name, item in results.items() if item.status == "ok"}
        if not successful:
            return results, ProviderResult(status="error", error="No provider returned an answer. Check API keys and provider errors above.")
        if len(successful) == 1:
            name, text = next(iter(successful.items()))
            return results, ProviderResult(status="ok", text=f"Single-provider result ({name}):\n\n{text}")
        synthesis = await self._synthesize(prompt, successful)
        return results, synthesis

    async def run_workflow(self, prompt: str, workflow: dict) -> tuple[list[dict], ProviderResult, int]:
        turns: list[dict] = []
        request_count = 0
        mode_instructions = {
            "respond": "Answer the user's task directly from your assigned perspective.",
            "critique": "Review the prior work. Identify consequential flaws and give precise corrections; do not recreate it independently.",
            "questions": "Ask only the few decisive questions needed to test or clarify the prior work. Do not answer them.",
            "audit": "Identify only consequential omissions, ignored constraints, and unresolved dependencies. Do not repeat prior points.",
            "synthesize": "Produce the final decision-ready answer by resolving the prior work. Do not summarize each agent separately.",
        }
        for step in workflow.get("steps", []):
            agent = dict(step)
            if not agent.get("enabled"):
                turns.append({**agent, "status": "skipped", "text": "", "error": "Agent is disabled."})
                continue
            prior = "\n\n".join(
                f"{turn['agent_name']} ({turn['mode']}):\n{turn['text']}"
                for turn in turns if turn["status"] == "ok"
            )
            visible_context = prior if agent.get("can_read_peers") else "(Peer messages are hidden for this agent.)"
            step_prompt = f"""USER TASK:
{prompt}

PRIOR WORKFLOW MESSAGES:
{visible_context or '(This is the first workflow step.)'}

CURRENT STEP — {agent['mode'].upper()}:
{mode_instructions[agent['mode']]}
Advance the work without unnecessary recap."""
            try:
                provider = self._provider_for_agent(agent)
                result = await self._safe_ask(provider, step_prompt, self._agent_system_prompt(agent))
            except ValueError as exc:
                result = ProviderResult(status="error", error=str(exc))
            request_count += 1
            turns.append({**agent, **vars(result)})

        successful = {f"{turn['position']}. {turn['agent_name']}": turn["text"] for turn in turns if turn["status"] == "ok"}
        synthesized_turn = next((turn for turn in reversed(turns) if turn["status"] == "ok" and turn["mode"] == "synthesize"), None)
        if synthesized_turn:
            synthesis = ProviderResult(status="ok", text=synthesized_turn["text"])
        elif successful:
            synthesis = await self._synthesize(prompt, successful)
            request_count += 1
        else:
            synthesis = ProviderResult(status="error", error="No enabled workflow agent returned an answer.")
        return turns, synthesis, request_count

    async def stream_workflow(self, prompt: str, workflow: dict) -> AsyncIterator[dict]:
        turns, request_count = [], 0
        mode_instructions = {
            "respond": "Answer the user's task directly from your assigned perspective.",
            "critique": "Review prior work, identify consequential flaws, and give precise corrections.",
            "questions": "Ask only the few decisive questions needed to test or clarify prior work.",
            "audit": "Identify only consequential omissions and unresolved dependencies.",
            "synthesize": "Produce the final decision-ready answer by resolving prior work.",
        }
        for step in workflow.get("steps", []):
            agent = dict(step)
            if not agent.get("enabled"):
                turn = {**agent, "status": "skipped", "text": "", "error": "Agent is disabled."}; turns.append(turn)
                yield {"type": "step_done", "turn": turn}; continue
            prior = "\n\n".join(f"{t['agent_name']} ({t['mode']}):\n{t['text']}" for t in turns if t["status"] == "ok")
            visible = prior if agent.get("can_read_peers") else "(Peer messages are hidden for this agent.)"
            step_prompt = f"USER TASK:\n{prompt}\n\nPRIOR WORKFLOW MESSAGES:\n{visible or '(This is the first workflow step.)'}\n\nCURRENT STEP — {agent['mode'].upper()}:\n{mode_instructions[agent['mode']]}\nAdvance the work without unnecessary recap."
            yield {"type": "step_start", "position": agent["position"], "agent_name": agent["agent_name"]}
            text, usage = "", {"input_tokens": 0, "output_tokens": 0}
            try:
                provider = self._provider_for_agent(agent)
                async for event in provider.stream(step_prompt, self._agent_system_prompt(agent)):
                    if event["type"] == "delta": text += event["text"]; yield {"type": "step_delta", "position": agent["position"], "text": event["text"]}
                    elif event["type"] == "usage": usage = {"input_tokens": event.get("input_tokens", 0), "output_tokens": event.get("output_tokens", 0)}
                turn = {**agent, "status": "ok", "text": text, "error": "", "usage_provider": provider.provider_id, "model": provider.model, **usage}
            except Exception as exc:
                turn = {**agent, "status": "error", "text": text, "error": str(exc), "usage_provider": agent.get("provider", ""), "model": agent.get("model", ""), **usage}
            request_count += 1; turns.append(turn); yield {"type": "step_done", "turn": turn}
        successful = {f"{t['position']}. {t['agent_name']}": t["text"] for t in turns if t["status"] == "ok"}
        synthesized = next((t for t in reversed(turns) if t["status"] == "ok" and t["mode"] == "synthesize"), None)
        synthesis = ProviderResult(status="ok", text=synthesized["text"]) if synthesized else await self._synthesize(prompt, successful) if successful else ProviderResult(status="error", error="No enabled workflow agent returned an answer.")
        if successful and not synthesized: request_count += 1
        yield {"type": "synthesis", "result": vars(synthesis)}
        yield {"type": "complete", "turns": turns, "synthesis": vars(synthesis), "request_count": request_count}

    async def stream_discussion(self, prompt: str, rounds: int) -> AsyncIterator[dict]:
        turns, request_count = [], 0
        for name in ("openai", "deepseek", "jury"):
            provider = self.providers[name]; text = ""; usage = {"input_tokens": 0, "output_tokens": 0}
            yield {"type": "turn_start", "round": 1, "provider": name}
            try:
                async for event in provider.stream(f"Task from the user:\n{prompt}\n\nGive your independent first response. Stay strictly inside your assigned role.", self._role_prompt(name)):
                    if event["type"] == "delta": text += event["text"]; yield {"type": "turn_delta", "round": 1, "provider": name, "text": event["text"]}
                    elif event["type"] == "usage": usage = {"input_tokens": event.get("input_tokens", 0), "output_tokens": event.get("output_tokens", 0)}
                turn = {"round": 1, "provider": name, "status": "ok", "text": text, "error": "", "usage_provider": provider.provider_id, "model": provider.model, **usage}
            except Exception as exc: turn = {"round": 1, "provider": name, "status": "error", "text": text, "error": str(exc), "usage_provider": provider.provider_id, "model": provider.model, **usage}
            turns.append(turn); request_count += 1; yield {"type": "turn_done", "turn": turn}
        for round_number in range(2, rounds + 2):
            snapshot = list(turns)
            for name in ("openai", "deepseek", "jury"):
                provider = self.providers[name]; own, peers = self._discussion_history(snapshot, name); text = ""; usage = {"input_tokens": 0, "output_tokens": 0}
                prompt_for_turn = f"Original task:\n{prompt}\n\nYOUR PREVIOUS MESSAGES:\n{own or '(none)'}\n\nOTHER TEAM MEMBERS' MESSAGES:\n{peers or '(none)'}\n\nThis is discussion round {round_number}. State only what you retain or revise, then address the most important peer point."
                yield {"type": "turn_start", "round": round_number, "provider": name}
                try:
                    async for event in provider.stream(prompt_for_turn, self._role_prompt(name)):
                        if event["type"] == "delta": text += event["text"]; yield {"type": "turn_delta", "round": round_number, "provider": name, "text": event["text"]}
                        elif event["type"] == "usage": usage = {"input_tokens": event.get("input_tokens", 0), "output_tokens": event.get("output_tokens", 0)}
                    turn = {"round": round_number, "provider": name, "status": "ok", "text": text, "error": "", "usage_provider": provider.provider_id, "model": provider.model, **usage}
                except Exception as exc: turn = {"round": round_number, "provider": name, "status": "error", "text": text, "error": str(exc), "usage_provider": provider.provider_id, "model": provider.model, **usage}
                turns.append(turn); request_count += 1; yield {"type": "turn_done", "turn": turn}
        successful = {f"round {t['round']} — {t['provider']}": t["text"] for t in turns if t["status"] == "ok"}
        synthesis = await self._synthesize(prompt, successful) if successful else ProviderResult(status="error", error="No provider returned an answer.")
        if successful: request_count += 1
        yield {"type": "synthesis", "result": vars(synthesis)}
        yield {"type": "complete", "turns": turns, "synthesis": vars(synthesis), "request_count": request_count}

    async def _synthesize(self, prompt: str, answers: dict[str, str]) -> ProviderResult:
        provider_name = "openai" if self.synthesis_provider == "jury" else self.synthesis_provider
        provider = self.providers.get(provider_name)
        if not provider or not provider.configured:
            provider = next((p for p in self.providers.values() if p.configured), None)
        if not provider:
            return ProviderResult(status="error", error="No configured provider is available for synthesis.")
        source_text = "\n\n".join(f"## {name}\n{text}" for name, text in answers.items())
        synthesis_prompt = f"""Original question:\n{prompt}\n\nTeam answers:\n{source_text}\n\nProduce a final decision-ready answer of no more than 5 complete sentences. Lead with the recommendation, keep only complementary insights, resolve disagreements when evidence allows, and use the final sentence for any unresolved issue or next action. Do not summarize each model separately and never end mid-sentence."""
        return await self._safe_ask(provider, synthesis_prompt, SYSTEM_PROMPT)

    async def discuss(self, prompt: str, rounds: int) -> tuple[list[dict], ProviderResult, int]:
        turns: list[dict] = []
        request_count = 0
        opening_tasks = [
            self._safe_ask(provider, f"Task from the user:\n{prompt}\n\nGive your independent first response. Stay strictly inside your assigned role.", self._role_prompt(name))
            for name, provider in ((name, self.providers[name]) for name in ("openai", "deepseek", "jury"))
        ]
        opening = await asyncio.gather(*opening_tasks)
        request_count += len(opening_tasks)
        for name, result in zip(("openai", "deepseek", "jury"), opening):
            turns.append({"round": 1, "provider": name, **vars(result)})

        for round_number in range(2, rounds + 2):
            names, tasks = [], []
            for name, provider in ((name, self.providers[name]) for name in ("openai", "deepseek", "jury")):
                if not provider.configured:
                    continue
                own_history, peer_history = self._discussion_history(turns, name)
                discussion_prompt = f"""Original task:
{prompt}

YOUR PREVIOUS MESSAGES — you wrote these; preserve continuity and never refer to them as another model's claims:
{own_history or '(none)'}

OTHER TEAM MEMBERS' MESSAGES — these were written by your peers, not by you:
{peer_history or '(none)'}

This is discussion round {round_number}. Briefly state what you retain or revise from your own position, then respond only to the most important peer point. Advance the work; do not recap the whole discussion."""
                names.append(name)
                tasks.append(self._safe_ask(provider, discussion_prompt, self._role_prompt(name)))
            if not tasks:
                break
            replies = await asyncio.gather(*tasks)
            request_count += len(tasks)
            for name, result in zip(names, replies):
                turns.append({"round": round_number, "provider": name, **vars(result)})

        successful = {f"round {turn['round']} — {turn['provider']}": turn["text"] for turn in turns if turn["status"] == "ok"}
        if not successful:
            synthesis = ProviderResult(status="error", error="No provider returned an answer. Check API keys and provider errors.")
        else:
            synthesis = await self._synthesize(prompt, successful)
            request_count += 1
        return turns, synthesis, request_count

    async def chat(self, prompt: str, history: list[dict]) -> dict[str, ProviderResult]:
        tasks = []
        for name in ("openai", "deepseek"):
            transcript = []
            for turn in history[-12:]:
                transcript.append(f"User: {turn.get('user', '')}")
                previous = turn.get(name, "")
                if previous:
                    transcript.append(f"Your previous reply: {previous}")
            chat_prompt = f"""Conversation history with this user:
{chr(10).join(transcript) or '(new conversation)'}

New user message:
{prompt}

Reply directly while maintaining continuity with your own previous replies. Do not mention the other model unless the user asks."""
            tasks.append(self._safe_ask(self.providers[name], chat_prompt, self._role_prompt(name)))
        replies = await asyncio.gather(*tasks)
        return dict(zip(("openai", "deepseek"), replies))

    async def chat_reply(self, provider_name: str, prompt: str, history: list[dict]) -> ProviderResult:
        chat_prompt = self._chat_reply_prompt(provider_name, prompt, history)
        return await self._safe_ask(self.providers[provider_name], chat_prompt, self._role_prompt(provider_name))

    async def stream_chat_reply(self, provider_name: str, prompt: str, history: list[dict]) -> AsyncIterator[dict]:
        chat_prompt = self._chat_reply_prompt(provider_name, prompt, history)
        async for event in self.providers[provider_name].stream(chat_prompt, self._role_prompt(provider_name)):
            yield event

    def _chat_reply_prompt(self, provider_name: str, prompt: str, history: list[dict]) -> str:
        transcript = []
        visible_history = history[-30:]
        if visible_history and visible_history[-1].get("speaker") == "user" and visible_history[-1].get("content", "").strip() == prompt.strip():
            visible_history = visible_history[:-1]
        for message in visible_history:
            speaker = message.get("speaker", "unknown")
            label = {"user": "User", "openai": "ChatGPT", "deepseek": "DeepSeek"}.get(speaker, speaker)
            transcript.append(f"{label}: {message.get('content', '')}")
        peer = "DeepSeek" if provider_name == "openai" else "ChatGPT"
        return f"""This is a live group chat between the user, ChatGPT, and DeepSeek.

VISIBLE CHAT HISTORY — messages labeled {peer} were written by the other model, and you must read and respond to them when relevant:
{chr(10).join(transcript) or '(new conversation)'}

Current user message:
{prompt}

Reply as {self.labels[provider_name]}. Maintain continuity, acknowledge or challenge the other model's relevant point without recapping the chat, and stay within 5 complete sentences."""

    def _discussion_history(self, turns: list[dict], current_provider: str) -> tuple[str, str]:
        own = []
        peers = []
        for turn in turns:
            if turn["status"] != "ok":
                continue
            entry = f"Round {turn['round']} — {self.labels[turn['provider']]}:\n{turn['text']}"
            (own if turn["provider"] == current_provider else peers).append(entry)
        return "\n\n".join(own)[-20_000:], "\n\n".join(peers)[-40_000:]

    def _role_prompt(self, name: str) -> str:
        return f"{SYSTEM_PROMPT}\n\n{self.roles[name]}"
