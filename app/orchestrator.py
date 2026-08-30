import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from .config import Settings
from .providers import AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAICompatibleProvider, OpenAIProvider, Provider, SYSTEM_PROMPT


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
            "gemini": GeminiProvider(settings.gemini_api_key, settings.gemini_model, timeout, settings.max_output_tokens),
            "anthropic": AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model, timeout, settings.max_output_tokens),
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
        elif agent["provider"] == "gemini":
            provider = GeminiProvider(self.settings.gemini_api_key, agent["model"], self.settings.request_timeout_seconds, self.settings.max_output_tokens)
        elif agent["provider"] == "anthropic":
            provider = AnthropicProvider(self.settings.anthropic_api_key, agent["model"], self.settings.request_timeout_seconds, self.settings.max_output_tokens)
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

    async def stream_debate(self, question: str, participants: list[dict], jury, debate_format: str, intervention: str = "", moderator_context=None, evidence_policy: str = "open", benchmark: bool = False, auto_stop_on_convergence: bool = True) -> AsyncIterator[dict]:
        turns, request_count = [], 0
        juries = jury if isinstance(jury, list) else [jury]
        evidence_rules = {
            "open": "You may use supplied evidence and general knowledge. Never fabricate citations.",
            "cite_facts": "Cite supplied evidence with exact [D#C#] labels for every material factual claim; clearly label unsupported inference.",
            "sources_only": "Use only supplied evidence. Every material factual claim must include an exact [D#C#] citation; say when evidence is insufficient.",
        }
        baseline = None
        convergence_check = None
        if benchmark:
            baseline_agent={"agent_name":"Baseline","provider":participants[0]["provider"],"model":participants[0]["model"],"role":"Independent decision maker","instructions":"Answer without seeing any debate.","max_sentences":5}
            baseline_provider=self._provider_for_agent(baseline_agent)
            baseline_result=await self._safe_ask(baseline_provider,f"QUESTION:\n{question}\n\nGive the best independent answer. {evidence_rules[evidence_policy]}",self._agent_system_prompt(baseline_agent));request_count+=1
            baseline={"status":baseline_result.status,"text":baseline_result.text,"error":baseline_result.error,"provider":participants[0]["provider"],"model":participants[0]["model"],"usage_provider":baseline_result.usage_provider,"input_tokens":baseline_result.input_tokens,"output_tokens":baseline_result.output_tokens}
            yield {"type":"baseline","baseline":baseline}
        stages = ["opening", "cross_examination", "rebuttal", "closing"]
        format_rules = {
            "adversarial": "Challenge opposing claims directly and concede only when the argument warrants it.",
            "decision": "Compare alternatives against feasibility, risk, evidence, and the user's desired outcome.",
            "socratic": "Expose assumptions through precise questions and reason from the answers.",
        }
        stage_rules = {
            "opening": "Present your strongest position independently: thesis, reasoning, and the evidence required to support it.",
            "cross_examination": "Ask one precise question of the named opponent that targets their weakest consequential claim. Do not answer the question yourself.",
            "rebuttal": "Answer questions directed at you, challenge the strongest opposing claim, and explicitly identify any point you now concede.",
            "closing": "State your final recommendation, what changed in your position, and the single most important remaining uncertainty.",
        }
        for stage_index, stage in enumerate(stages, start=1):
            live_intervention = await moderator_context() if moderator_context else intervention
            snapshot=list(turns)
            for index, participant in enumerate(participants):
                own="\n\n".join(f"{t['stage']}: {t['text']}" for t in snapshot if t["participant_id"]==participant["id"] and t["status"]=="ok")
                peers="\n\n".join(f"{t['name']} ({t['position']}) — {t['stage']}: {t['text']}" for t in snapshot if t["participant_id"]!=participant["id"] and t["status"]=="ok")
                target=participants[(index+1)%len(participants)]
                target_instruction=f"Address your question specifically to {target['name']} ({target['position']})." if stage=="cross_examination" else ""
                prompt=f"""DEBATE QUESTION:\n{question}\n\nDEBATE FORMAT:\n{debate_format}: {format_rules[debate_format]}\n\nEVIDENCE POLICY:\n{evidence_rules[evidence_policy]}\n\nYOUR IDENTITY:\nYou are {participant['name']}. Your assigned position is: {participant['position']}\n\nYOUR OWN PREVIOUS STATEMENTS — you wrote these; never attribute them to an opponent:\n{own or '(none yet)'}\n\nOPPONENT STATEMENTS — written by other participants:\n{peers or '(none yet)'}\n\nUSER INTERVENTION:\n{live_intervention or '(none)'}\n\nCURRENT STAGE — {stage.replace('_',' ').upper()}:\n{stage_rules[stage]} {target_instruction}\nAdvance the debate without recap."""
                agent={"agent_name":participant["name"],"provider":participant["provider"],"model":participant["model"],"role":f"Debater assigned to defend: {participant['position']}","instructions":format_rules[debate_format],"max_sentences":5}
                yield {"type":"turn_start","stage":stage,"stage_index":stage_index,"participant":participant}
                text="";usage={"input_tokens":0,"output_tokens":0}
                try:
                    provider=self._provider_for_agent(agent)
                    async for event in provider.stream(prompt,self._agent_system_prompt(agent)):
                        if event["type"]=="delta":text+=event["text"];yield {"type":"turn_delta","stage":stage,"participant_id":participant["id"],"text":event["text"]}
                        elif event["type"]=="usage":usage={"input_tokens":event.get("input_tokens",0),"output_tokens":event.get("output_tokens",0)}
                    turn={"stage":stage,"stage_index":stage_index,"participant_id":participant["id"],"name":participant["name"],"position":participant["position"],"provider":participant["provider"],"model":participant["model"],"status":"ok","text":text,"error":"","usage_provider":provider.provider_id,**usage}
                except Exception as exc:
                    turn={"stage":stage,"stage_index":stage_index,"participant_id":participant["id"],"name":participant["name"],"position":participant["position"],"provider":participant["provider"],"model":participant["model"],"status":"error","text":text,"error":str(exc),"usage_provider":participant["provider"],**usage}
                turns.append(turn);request_count+=1;yield {"type":"turn_done","turn":turn}
            if stage=="rebuttal" and auto_stop_on_convergence:
                blind={participant["id"]:f"P{idx+1}" for idx,participant in enumerate(participants)}
                checkpoint="\n\n".join(f"[{turn['stage']}] {blind[turn['participant_id']]}: {turn['text']}" for turn in turns if turn["status"]=="ok")
                convergence_agent={"agent_name":"Convergence Monitor","provider":juries[0]["provider"],"model":juries[0]["model"],"role":"Blind debate convergence monitor","instructions":"Stop only when the material recommendation and reasoning have converged.","max_sentences":2}
                convergence_provider=self._provider_for_agent(convergence_agent);check=await self._safe_ask(convergence_provider,f"QUESTION:\n{question}\n\nANONYMIZED DEBATE:\n{checkpoint[-60000:]}\n\nReturn only JSON: {{\"converged\":true|false,\"reason\":\"...\"}}",self._agent_system_prompt(convergence_agent));request_count+=1
                try:
                    candidate=check.text.strip();match=re.search(r"```(?:json)?\s*(\{.*\})\s*```",candidate,re.S);convergence=json.loads(match.group(1) if match else candidate) if check.status=="ok" else {"converged":False,"reason":check.error}
                except Exception:convergence={"converged":False,"reason":"Convergence check was inconclusive."}
                convergence["usage"]={"usage_provider":check.usage_provider,"model":juries[0]["model"],"input_tokens":check.input_tokens,"output_tokens":check.output_tokens}
                convergence_check=convergence
                yield {"type":"convergence","convergence":convergence}
                if convergence.get("converged"):break
        aliases={participant["id"]:f"P{index+1}" for index,participant in enumerate(participants)}
        reverse_aliases={value:key for key,value in aliases.items()}
        transcript="\n\n".join(f"[{t['stage']}] {aliases[t['participant_id']]}: {t['text'] or t['error']}" for t in turns)
        reports=[]
        for jury_index,jury_item in enumerate(juries,start=1):
            jury_agent={"agent_name":f"Independent Jury {jury_index}","provider":jury_item["provider"],"model":jury_item["model"],"role":"Impartial blind debate judge","instructions":"Participant identities and model providers are hidden. Judge only transcript quality.","max_sentences":5}
            baseline_block=f"\n\nINDEPENDENT BASELINE ANSWER:\n{baseline['text']}" if baseline and baseline.get("status")=="ok" else ""
            jury_prompt=f"""QUESTION:\n{question}\n\nANONYMIZED DEBATE TRANSCRIPT:\n{transcript[-90000:]}{baseline_block}\n\nReturn only valid JSON with this schema: {{\"verdict\":\"concise decision\",\"winner_id\":\"P1, P2, ... or tie\",\"strongest_argument\":\"...\",\"converged\":false,\"consensus\":[\"...\"],\"disagreements\":[{{\"claim\":\"...\",\"positions\":\"...\"}}],\"unresolved_questions\":[\"...\"],\"claims\":[{{\"claim_id\":\"C1\",\"participant_id\":\"P1\",\"claim\":\"...\",\"citations\":[\"D1C1\"],\"status\":\"supported|challenged|conceded|unresolved\",\"challenged_by\":[\"P2\"]}}],\"scores\":[{{\"participant_id\":\"P1\",\"reasoning\":0,\"evidence\":0,\"responsiveness\":0,\"consistency\":0,\"total\":0,\"justification\":\"...\"}}],\"debate_vs_baseline\":{{\"winner\":\"debate|baseline|tie|not_run\",\"reason\":\"...\"}}}}. Score each dimension 0-10 and total 0-40. Base every score only on the anonymized text; never infer model identity."""
            yield {"type":"jury_start","jury_index":jury_index,"jury":{"label":f"Jury {jury_index}"}}
            try:
                jury_provider=self._provider_for_agent(jury_agent);result=await self._safe_ask(jury_provider,jury_prompt,self._agent_system_prompt(jury_agent));request_count+=1
                if result.status!="ok":raise ValueError(result.error)
                candidate=result.text.strip();match=re.search(r"```(?:json)?\s*(\{.*\})\s*```",candidate,re.S);item=json.loads(match.group(1) if match else candidate)
                item["winner_id"]=reverse_aliases.get(item.get("winner_id"),item.get("winner_id","tie"))
                for score in item.get("scores",[]):score["participant_id"]=reverse_aliases.get(score.get("participant_id"),score.get("participant_id"))
                for claim in item.get("claims",[]):
                    claim["participant_id"]=reverse_aliases.get(claim.get("participant_id"),claim.get("participant_id"));claim["challenged_by"]=[reverse_aliases.get(value,value) for value in claim.get("challenged_by",[])]
                item["jury"]={**jury_item,"usage_provider":result.usage_provider,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens};reports.append(item)
            except Exception as exc:reports.append({"verdict":"Jury failed.","winner_id":"tie","consensus":[],"disagreements":[],"unresolved_questions":[str(exc)],"claims":[],"scores":[],"jury":jury_item})
        report=self._aggregate_juries(reports,participants)
        report["citation_audit"]=self._citation_audit(turns,question,evidence_policy)
        report["baseline"]=baseline
        report["convergence_check"]=convergence_check
        if convergence_check and convergence_check.get("converged"):report["converged"]=True
        report["blind_jury_count"]=len(juries)
        report["jury_reports"]=reports
        yield {"type":"report","report":report}
        yield {"type":"complete","turns":turns,"report":report,"request_count":request_count}

    @staticmethod
    def _citation_audit(turns: list[dict], question: str, evidence_policy: str):
        valid=set(re.findall(r"\[?(D\d+C\d+)\]?",question));by_participant={}
        for turn in turns:
            if turn.get("status")!="ok":continue
            found=set(re.findall(r"\[?(D\d+C\d+)\]?",turn.get("text","")));entry=by_participant.setdefault(turn["participant_id"],{"valid":set(),"invalid":set(),"turns":0,"cited_turns":0})
            entry["turns"]+=1;entry["valid"].update(found&valid);entry["invalid"].update(found-valid);entry["cited_turns"]+=bool(found&valid)
        return {"policy":evidence_policy,"available_citations":sorted(valid),"participants":{key:{"valid":sorted(value["valid"]),"invalid":sorted(value["invalid"]),"coverage":round(value["cited_turns"]/max(1,value["turns"]),2),"passes":not value["invalid"] and (evidence_policy=="open" or value["cited_turns"]==value["turns"])} for key,value in by_participant.items()}}

    @staticmethod
    def _aggregate_juries(reports: list[dict], participants: list[dict]):
        totals={participant["id"]:[] for participant in participants};dimensions={participant["id"]:{name:[] for name in ("reasoning","evidence","responsiveness","consistency")} for participant in participants}
        for report in reports:
            for score in report.get("scores",[]):
                participant_id=score.get("participant_id")
                if participant_id not in totals:continue
                totals[participant_id].append(float(score.get("total",0)))
                for name in dimensions[participant_id]:dimensions[participant_id][name].append(float(score.get(name,0)))
        scores=[]
        for participant in participants:
            participant_id=participant["id"];average=lambda values:round(sum(values)/len(values),1) if values else 0
            scores.append({"participant_id":participant_id,"total":average(totals[participant_id]),**{name:average(values) for name,values in dimensions[participant_id].items()},"justification":f"Average of {len(totals[participant_id])} blind jury score(s)."})
        ranked=sorted(scores,key=lambda item:item["total"],reverse=True);winner="tie" if len(ranked)>1 and abs(ranked[0]["total"]-ranked[1]["total"])<0.5 else ranked[0]["participant_id"] if ranked else "tie"
        votes=[report.get("winner_id","tie") for report in reports];agreement=round(max((votes.count(value) for value in set(votes)),default=0)/max(1,len(votes)),2)
        primary=reports[0] if reports else {}
        return {"verdict":primary.get("verdict","No valid jury verdict."),"winner_id":winner,"strongest_argument":primary.get("strongest_argument",""),"converged":all(report.get("converged",False) for report in reports) if reports else False,"consensus":primary.get("consensus",[]),"disagreements":primary.get("disagreements",[]),"unresolved_questions":primary.get("unresolved_questions",[]),"claims":primary.get("claims",[]),"scores":scores,"jury_agreement":agreement,"debate_vs_baseline":primary.get("debate_vs_baseline",{"winner":"not_run","reason":""})}

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
