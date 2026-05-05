"""
🧠 Conselho Multi-Agente v5 — debate oculto, resposta final em destaque
==========================================================================
Correções da v4:
1. Grok Multi-Agent agora usa Responses API (não Chat Completions)
2. Claude usa streaming (necessário pra requests longas com effort alto)
3. UI: resposta final em destaque, debate fica oculto atrás de expander
"""

import streamlit as st
import asyncio
import os
import time
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

try:
    from google import genai as google_genai
    from google.genai import types as gemini_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="Conselho Multi-Agente v5", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 16px; color: white;
        margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    .captain-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.18) 0%, rgba(118,75,162,0.18) 100%);
        border: 2px solid rgba(102,126,234,0.5);
        padding: 2.5rem; border-radius: 20px; margin: 1rem 0;
        box-shadow: 0 15px 50px rgba(102,126,234,0.25);
    }
    .captain-card h2 { margin-top: 0; }
    .cost-card {
        padding: 1rem; border-radius: 12px; margin: 1rem 0;
    }
    .stage-pill {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.85em; margin-right: 8px;
    }
    footer { visibility: hidden; } #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CATÁLOGO DE MODELOS / PRICING
# ============================================================

CLAUDE_MODELS = {
    "claude-opus-4-7":   {"label": "Claude Opus 4.7 (top)",      "in":  5.0, "out": 25.0},
    "claude-opus-4-6":   {"label": "Claude Opus 4.6",             "in":  5.0, "out": 25.0},
    "claude-sonnet-4-6": {"label": "Claude Sonnet 4.6 (barato)",  "in":  3.0, "out": 15.0},
}

GPT_MODELS = {
    "gpt-5.5":    {"label": "GPT-5.5 (top)",         "in": 1.75, "out": 14.0},
    "gpt-5.4":    {"label": "GPT-5.4",                "in": 1.50, "out": 12.0},
    "gpt-5.2":    {"label": "GPT-5.2",                "in": 1.50, "out": 12.0},
    "gpt-4o":     {"label": "GPT-4o (sem thinking)",  "in": 2.50, "out": 10.0},
}

GROK_MODELS = {
    "grok-4.20-multi-agent":         {"label": "Grok 4.2 Multi-Agent (top)", "in": 2.0, "out": 6.0, "api": "responses"},
    "grok-4.20-beta-0309-reasoning": {"label": "Grok 4.2 Reasoning",          "in": 1.5, "out": 5.0, "api": "chat"},
    "grok-4-fast":                   {"label": "Grok 4 Fast (barato)",        "in": 0.5, "out": 2.0, "api": "chat"},
}

GEMINI_MODELS = {
    "gemini-3.1-pro-preview": {"label": "Gemini 3.1 Pro (top, Deep Think Mini)", "in": 2.0,  "out": 12.0},
    "gemini-3-pro":           {"label": "Gemini 3 Pro",                          "in": 1.5,  "out": 10.0},
    "gemini-2.5-pro":         {"label": "Gemini 2.5 Pro",                        "in": 1.25, "out": 10.0},
}

KIMI_MODELS = {
    "kimi-k2.6": {"label": "Kimi K2.6 (top)", "in": 0.74, "out": 3.49},
    "kimi-k2.5": {"label": "Kimi K2.5",       "in": 0.50, "out": 2.50},
}

EFFORT_MULTIPLIERS = {
    "none": 0.5, "low": 1.0, "medium": 2.0, "high": 4.0, "xhigh": 7.0, "max": 10.0,
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuração")

    st.subheader("⚡ Presets rápidos")
    preset_cols = st.columns(3)
    if preset_cols[0].button("🪙 Econômico", use_container_width=True):
        st.session_state["preset"] = "economy"
    if preset_cols[1].button("⚖️ Balanceado", use_container_width=True):
        st.session_state["preset"] = "balanced"
    if preset_cols[2].button("🚀 Top tier", use_container_width=True):
        st.session_state["preset"] = "top"
    preset = st.session_state.get("preset", None)

    def preset_val(economy, balanced, top, default):
        if preset == "economy":  return economy
        if preset == "balanced": return balanced
        if preset == "top":      return top
        return default

    st.divider()

    with st.expander("🔑 API Keys", expanded=not bool(os.getenv("ANTHROPIC_API_KEY"))):
        anthropic_key = st.text_input("Anthropic", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        openai_key = st.text_input("OpenAI", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        xai_key = st.text_input("xAI (Grok)", type="password", value=os.getenv("XAI_API_KEY", ""))
        gemini_key = st.text_input("Google (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        kimi_key = st.text_input("Moonshot (Kimi)", type="password", value=os.getenv("MOONSHOT_API_KEY", ""))

    st.divider()
    st.subheader("🤖 Agentes")

    with st.expander("🔍 Harper · Grok", expanded=True):
        enable_harper = st.checkbox("Ativar", value=True, key="en_harper")
        harper_model = st.selectbox("Modelo", list(GROK_MODELS.keys()),
            index=preset_val(2, 1, 0, 1),  # Top usa multi-agent, Balanceado usa reasoning
            format_func=lambda k: GROK_MODELS[k]["label"], key="m_harper")
        if GROK_MODELS[harper_model]["api"] == "responses":
            st.caption("ℹ️ Multi-Agent usa Responses API (4 cabeças internas)")
        else:
            st.caption("ℹ️ Reasoning embutido no modelo")

    with st.expander("🧠 Benjamin · Claude (lógica)", expanded=True):
        enable_benjamin = st.checkbox("Ativar", value=True, key="en_benjamin")
        benjamin_model = st.selectbox("Modelo", list(CLAUDE_MODELS.keys()),
            index=preset_val(2, 0, 0, 0),
            format_func=lambda k: CLAUDE_MODELS[k]["label"], key="m_benjamin")
        benjamin_effort = st.select_slider("Effort",
            options=["low", "medium", "high", "xhigh", "max"],
            value=preset_val("low", "medium", "max", "high"), key="e_benjamin")

    with st.expander("⚖️ Aria · Claude (nuance)"):
        enable_aria = st.checkbox("Ativar", value=False, key="en_aria")
        aria_model = st.selectbox("Modelo", list(CLAUDE_MODELS.keys()), index=1,
            format_func=lambda k: CLAUDE_MODELS[k]["label"], key="m_aria")
        aria_effort = st.select_slider("Effort",
            options=["low", "medium", "high", "xhigh", "max"],
            value=preset_val("low", "medium", "max", "high"), key="e_aria")

    with st.expander("🎨 Lucas · GPT (criatividade)", expanded=True):
        enable_lucas = st.checkbox("Ativar", value=True, key="en_lucas")
        lucas_model = st.selectbox("Modelo", list(GPT_MODELS.keys()),
            index=preset_val(3, 0, 0, 0),
            format_func=lambda k: GPT_MODELS[k]["label"], key="m_lucas")
        if lucas_model != "gpt-4o":
            lucas_effort = st.select_slider("Reasoning effort",
                options=["none", "low", "medium", "high", "xhigh"],
                value=preset_val("low", "medium", "xhigh", "high"), key="e_lucas")
        else:
            lucas_effort = "none"
            st.caption("ℹ️ GPT-4o não tem reasoning")

    with st.expander("📚 Sage · Gemini"):
        enable_sage = st.checkbox("Ativar", value=False, key="en_sage")
        sage_model = st.selectbox("Modelo", list(GEMINI_MODELS.keys()), index=0,
            format_func=lambda k: GEMINI_MODELS[k]["label"], key="m_sage")
        sage_thinking = st.select_slider("Thinking level",
            options=["low", "medium", "high"],
            value=preset_val("low", "medium", "high", "medium"), key="t_sage")

    with st.expander("⚡ Kai · Kimi"):
        enable_kai = st.checkbox("Ativar", value=False, key="en_kai")
        kai_model = st.selectbox("Modelo", list(KIMI_MODELS.keys()), index=0,
            format_func=lambda k: KIMI_MODELS[k]["label"], key="m_kai")
        kai_thinking = st.toggle("Thinking ON", value=True, key="t_kai")

    st.divider()
    st.subheader("🎯 Captain")
    captain_model = st.selectbox("Modelo", list(CLAUDE_MODELS.keys()),
        index=preset_val(2, 1, 0, 1),
        format_func=lambda k: CLAUDE_MODELS[k]["label"], key="m_captain")
    captain_effort = st.select_slider("Effort",
        options=["low", "medium", "high", "xhigh", "max"],
        value=preset_val("low", "medium", "max", "high"), key="e_captain")


def get_clients():
    clients = {
        "anthropic": AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None,
        "openai":    AsyncOpenAI(api_key=openai_key) if openai_key else None,
        "grok":      AsyncOpenAI(api_key=xai_key, base_url="https://api.x.ai/v1") if xai_key else None,
        "kimi":      AsyncOpenAI(api_key=kimi_key, base_url="https://api.moonshot.ai/v1") if kimi_key else None,
        "gemini":    None,
    }
    if gemini_key and GEMINI_AVAILABLE:
        clients["gemini"] = google_genai.Client(api_key=gemini_key)
    return clients


def extract_responses_text(resp):
    """Extrai texto da Responses API (OpenAI/xAI compatível)."""
    text = getattr(resp, "output_text", "") or ""
    if not text and hasattr(resp, "output"):
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", None) == "output_text":
                        text += c.text
    return text


# ============================================================
# AGENTES — fixados pra usar a API correta
# ============================================================

async def agent_harper(prompt, clients):
    """🔍 Grok — Multi-Agent via Responses API; outros via Chat Completions."""
    system_msg = ("Você é Harper, especialista em pesquisa factual. "
                  "Traga dados verificáveis, números, fontes. Responda em pt-BR.")

    if GROK_MODELS[harper_model]["api"] == "responses":
        resp = await clients["grok"].responses.create(
            model=harper_model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
        )
        return {"text": extract_responses_text(resp), "thinking": None}
    else:
        resp = await clients["grok"].chat.completions.create(
            model=harper_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
        )
        return {
            "text": resp.choices[0].message.content,
            "thinking": getattr(resp.choices[0].message, "reasoning_content", None)
        }


async def _claude_call_streaming(model, effort, system, prompt, client):
    """Claude com streaming — necessário pra requests longas (effort alto)."""
    kwargs = {
        "model": model,
        "max_tokens": 64000 if effort in ("xhigh", "max") else 32000,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model == "claude-opus-4-7":
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        kwargs["output_config"] = {"effort": effort}
    else:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}

    text, thinking = "", ""
    async with client.messages.stream(**kwargs) as stream:
        final = await stream.get_final_message()
        for block in final.content:
            if block.type == "thinking":
                thinking = (block.thinking or "") if hasattr(block, "thinking") else ""
            elif block.type == "text":
                text = block.text
    return {"text": text, "thinking": thinking}


async def agent_benjamin(prompt, clients):
    return await _claude_call_streaming(
        benjamin_model, benjamin_effort,
        ("Você é Benjamin, especialista em raciocínio lógico estrutural. "
         "Decomponha em premissas, identifique falhas, construa argumentos rigorosos. "
         "Responda em pt-BR."),
        prompt, clients["anthropic"]
    )


async def agent_aria(prompt, clients):
    return await _claude_call_streaming(
        aria_model, aria_effort,
        ("Você é Aria, especialista em nuance e tradeoffs. "
         "Pondere prós e contras, tensões entre objetivos, considerações éticas. "
         "Responda em pt-BR."),
        prompt, clients["anthropic"]
    )


async def agent_lucas(prompt, clients):
    if lucas_model == "gpt-4o":
        resp = await clients["openai"].chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Você é Lucas, criatividade e UX. Ângulos não-óbvios. Responda em pt-BR."
                )},
                {"role": "user", "content": prompt}
            ],
        )
        return {"text": resp.choices[0].message.content, "thinking": None}

    resp = await clients["openai"].responses.create(
        model=lucas_model,
        reasoning={"effort": lucas_effort},
        input=[
            {"role": "system", "content": (
                "Você é Lucas, especialista em perspectivas criativas e UX. "
                "Traga ângulos não-óbvios, questione premissas. Responda em pt-BR."
            )},
            {"role": "user", "content": prompt}
        ],
    )
    return {"text": extract_responses_text(resp), "thinking": None}


async def agent_sage(prompt, clients):
    full_prompt = (
        "Você é Sage, síntese ampla e conexões transversais. "
        "Leituras holísticas, padrões não-evidentes. Responda em pt-BR.\n\n"
        f"Pergunta: {prompt}"
    )
    config = (gemini_types.GenerateContentConfig(
        thinking_config=gemini_types.ThinkingConfig(thinking_level=sage_thinking)
    ) if sage_model.startswith("gemini-3") else None)

    resp = await asyncio.to_thread(
        clients["gemini"].models.generate_content,
        model=sage_model,
        contents=full_prompt,
        config=config,
    )
    return {"text": resp.text, "thinking": None}


async def agent_kai(prompt, clients):
    extra = {} if kai_thinking else {"thinking": {"type": "disabled"}}
    resp = await clients["kimi"].chat.completions.create(
        model=kai_model,
        messages=[
            {"role": "system", "content": (
                "Você é Kai, execução técnica concreta. Implementação prática, "
                "ferramentas, código, edge cases. Responda em pt-BR."
            )},
            {"role": "user", "content": prompt}
        ],
        extra_body=extra,
        max_tokens=16000,
    )
    return {
        "text": resp.choices[0].message.content,
        "thinking": getattr(resp.choices[0].message, "reasoning_content", None)
    }


# ============================================================
# REGISTRY
# ============================================================

ALL_AGENTS = [
    (lambda: enable_harper,   "Harper",   agent_harper,   "🔍",
        lambda: GROK_MODELS[harper_model]["label"],     "#00d4ff",
        lambda: GROK_MODELS[harper_model],    None),
    (lambda: enable_benjamin, "Benjamin", agent_benjamin, "🧠",
        lambda: f"{CLAUDE_MODELS[benjamin_model]['label']} · {benjamin_effort}", "#ff6b9d",
        lambda: CLAUDE_MODELS[benjamin_model], lambda: benjamin_effort),
    (lambda: enable_aria,     "Aria",     agent_aria,     "⚖️",
        lambda: f"{CLAUDE_MODELS[aria_model]['label']} · {aria_effort}",         "#ff9d6b",
        lambda: CLAUDE_MODELS[aria_model],     lambda: aria_effort),
    (lambda: enable_lucas,    "Lucas",    agent_lucas,    "🎨",
        lambda: f"{GPT_MODELS[lucas_model]['label']} · {lucas_effort}",          "#ffd93d",
        lambda: GPT_MODELS[lucas_model],       lambda: lucas_effort),
    (lambda: enable_sage,     "Sage",     agent_sage,     "📚",
        lambda: f"{GEMINI_MODELS[sage_model]['label']} · {sage_thinking}",       "#9d4edd",
        lambda: GEMINI_MODELS[sage_model],     lambda: sage_thinking),
    (lambda: enable_kai,      "Kai",      agent_kai,      "⚡",
        lambda: KIMI_MODELS[kai_model]["label"],         "#06ffa5",
        lambda: KIMI_MODELS[kai_model],        None),
]


def active_agents():
    return [(name, fn, emoji, label_fn, color, pricing_fn, effort_fn)
            for check, name, fn, emoji, label_fn, color, pricing_fn, effort_fn in ALL_AGENTS
            if check()]


# ============================================================
# CUSTO
# ============================================================

def estimate_cost(agents, captain_pricing, captain_eff):
    INPUT_TOKENS = 2000
    INPUT_DEBATE = 8000
    total = 0.0
    breakdown = []

    for name, _, emoji, _, _, pricing_fn, effort_fn in agents:
        pricing = pricing_fn()
        effort = effort_fn() if effort_fn else "medium"
        mult = EFFORT_MULTIPLIERS.get(effort, 2.0)
        out = 1500 * mult
        cost_1 = (INPUT_TOKENS * pricing["in"] + out * pricing["out"]) / 1_000_000
        cost_2 = (INPUT_DEBATE * pricing["in"] + out * pricing["out"]) / 1_000_000
        agent_total = cost_1 + cost_2
        total += agent_total
        breakdown.append((f"{emoji} {name}", agent_total))

    captain_input = INPUT_DEBATE * 2
    captain_mult = EFFORT_MULTIPLIERS.get(captain_eff, 2.0)
    captain_out = 2000 * captain_mult
    captain_cost = (captain_input * captain_pricing["in"] + captain_out * captain_pricing["out"]) / 1_000_000
    total += captain_cost
    breakdown.append(("🎯 Captain", captain_cost))
    return total, breakdown


# ============================================================
# CRÍTICA + CAPTAIN
# ============================================================

def build_critique_prompt(agent_name, question, all_outputs):
    others = "\n\n".join(
        f"### {n}:\n{o['text']}" for n, o in all_outputs.items() if n != agent_name
    )
    return (
        f"PERGUNTA ORIGINAL:\n{question}\n\n"
        f"RESPOSTAS DOS OUTROS AGENTES:\n{others}\n\n"
        f"Você é {agent_name}. Identifique erros, contradições ou pontos fracos. "
        f"Refine sua posição. Seja direto."
    )


async def captain_synthesize(question, initial, critiques, clients, agents_meta):
    debate = ""
    for name, _, emoji, label_fn, _, _, _ in agents_meta:
        debate += (
            f"\n\n=== {emoji} {name} ({label_fn()}) ===\n"
            f"[Inicial]\n{initial[name]['text']}\n\n"
            f"[Refinamento]\n{critiques[name]['text']}"
        )
    agent_list = ", ".join(name for name, _, _, _, _, _, _ in agents_meta)

    kwargs = {
        "model": captain_model,
        "max_tokens": 64000 if captain_effort in ("xhigh", "max") else 32000,
        "system": (
            f"Você é o Captain. Integre as perspectivas dos agentes ({agent_list}) "
            "que debateram em duas rodadas. Resolva contradições explicitamente, "
            "descarte argumentos fracos, tome posição clara. "
            "Estruture em markdown: TL;DR no topo, depois análise, depois recomendação. "
            "Responda em pt-BR."
        ),
        "messages": [{
            "role": "user",
            "content": f"PERGUNTA:\n{question}\n\nDEBATE:{debate}\n\nProduza a resposta final integrada."
        }],
    }
    if captain_model == "claude-opus-4-7":
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        kwargs["output_config"] = {"effort": captain_effort}
    else:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": captain_effort}

    text, thinking = "", ""
    async with clients["anthropic"].messages.stream(**kwargs) as stream:
        final = await stream.get_final_message()
        for block in final.content:
            if block.type == "thinking":
                thinking = (block.thinking or "") if hasattr(block, "thinking") else ""
            elif block.type == "text":
                text = block.text
    return {"text": text, "thinking": thinking}


# ============================================================
# ORQUESTRAÇÃO — UI focada na resposta final
# ============================================================

async def run_council(question, ui_slots, agents):
    clients = get_clients()
    n = len(agents)
    t0 = time.time()

    # ── Fase 1 ──
    ui_slots["status"].info(f"💭 **Fase 1/3** — {n} agentes pensando em paralelo... (pode demorar alguns minutos)")
    results_1 = await asyncio.gather(
        *[fn(question, clients) for _, fn, _, _, _, _, _ in agents],
        return_exceptions=True
    )
    initial = {}
    errors = []
    for (name, _, _, _, _, _, _), r in zip(agents, results_1):
        if isinstance(r, Exception):
            initial[name] = {"text": f"⚠️ Erro: {type(r).__name__}: {str(r)[:300]}", "thinking": None}
            errors.append((name, str(r)))
        else:
            initial[name] = r
    ui_slots["status"].info(f"💬 **Fase 2/3** — Agentes se criticando mutuamente... ({time.time()-t0:.0f}s)")

    # ── Fase 2: crítica ──
    results_2 = await asyncio.gather(
        *[fn(build_critique_prompt(name, question, initial), clients)
          for name, fn, _, _, _, _, _ in agents],
        return_exceptions=True
    )
    critiques = {}
    for (name, _, _, _, _, _, _), r in zip(agents, results_2):
        if isinstance(r, Exception):
            critiques[name] = {"text": f"⚠️ Erro: {type(r).__name__}: {str(r)[:300]}", "thinking": None}
        else:
            critiques[name] = r

    # ── Fase 3: Captain ──
    ui_slots["status"].info(f"🎯 **Fase 3/3** — Captain compilando resposta final... ({time.time()-t0:.0f}s)")
    final = await captain_synthesize(question, initial, critiques, clients, agents)

    elapsed = time.time() - t0
    ui_slots["status"].empty()

    # ── RESPOSTA FINAL EM DESTAQUE ──
    with ui_slots["final"]:
        st.markdown('<div class="captain-card">', unsafe_allow_html=True)
        meta = f"⏱️ {elapsed:.0f}s · 🤖 {n} agentes · 2 rodadas de debate"
        if errors:
            meta += f" · ⚠️ {len(errors)} agente(s) com erro"
        st.caption(meta)
        st.markdown("## 🎯 Resposta do Conselho")
        st.markdown(final["text"])

        if final.get("thinking"):
            with st.expander("💭 Raciocínio do Captain"):
                st.caption(final["thinking"][:8000])

        st.markdown('</div>', unsafe_allow_html=True)

    # ── DEBATE COMPLETO (oculto por padrão) ──
    with ui_slots["debate"]:
        with st.expander(f"🎭 Ver o debate completo ({n} agentes, 2 rodadas)", expanded=False):
            st.caption("As respostas individuais e os refinamentos após a crítica cruzada.")
            st.divider()

            for name, _, emoji, label_fn, color, _, _ in agents:
                st.markdown(
                    f"<div style='border-left: 4px solid {color}; padding: 8px 16px; margin: 12px 0;'>"
                    f"<b>{emoji} {name}</b> · <span style='opacity:0.7'>{label_fn()}</span></div>",
                    unsafe_allow_html=True
                )
                tab1, tab2 = st.tabs(["📝 Resposta inicial", "🔄 Refinamento após debate"])
                with tab1:
                    if initial[name].get("thinking"):
                        with st.expander("💭 thinking"):
                            st.caption(initial[name]["thinking"][:4000])
                    st.markdown(initial[name]["text"])
                with tab2:
                    st.markdown(critiques[name]["text"])
                st.divider()


# ============================================================
# UI PRINCIPAL
# ============================================================

agents = active_agents()

st.markdown(f"""
<div class="main-header">
    <h1>🧠 Conselho Multi-Agente</h1>
    <p>{len(agents)} agentes · debate paralelo + crítica cruzada + síntese final</p>
</div>
""", unsafe_allow_html=True)

# Estimador
if len(agents) >= 2:
    cost, breakdown = estimate_cost(agents, CLAUDE_MODELS[captain_model], captain_effort)
    cost_low = cost * 0.4
    cost_high = cost * 2.5
    color = "#22c55e" if cost < 0.5 else "#f59e0b" if cost < 2.0 else "#ef4444"
    st.markdown(f"""
    <div class="cost-card" style="border: 1px solid {color}; background: {color}15;">
        <b style="color:{color}">💰 Custo estimado: ${cost_low:.2f} – ${cost_high:.2f} por pergunta</b>
        <small style="opacity:0.7"> · estimativa central ${cost:.2f}</small>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Breakdown por agente"):
        for name, c in breakdown:
            st.write(f"{name}: ${c:.3f}")

if len(agents) == 0:
    st.warning("⚠️ Ative pelo menos 1 agente na sidebar.")
elif len(agents) == 1:
    st.info("ℹ️ Com 1 agente só não tem debate. Ative pelo menos 2.")
else:
    cols_meta = st.columns(len(agents))
    for (name, _, emoji, label_fn, color, _, _), col in zip(agents, cols_meta):
        with col:
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
                f"<b>{emoji} {name}</b><br><small style='opacity:0.7'>{label_fn()}</small></div>",
                unsafe_allow_html=True
            )

st.write("")
question = st.text_area("Sua pergunta:",
    placeholder="Ex: Qual a melhor arquitetura pra sincronizar HubSpot, Salesforce e NetSuite?",
    height=100)

col_a, _ = st.columns([1, 4])
with col_a:
    go = st.button("🚀 Convocar conselho", type="primary",
                   use_container_width=True, disabled=len(agents) < 2)

# Validação
missing = []
if not anthropic_key: missing.append("Anthropic")
if enable_harper and not xai_key: missing.append("xAI")
if enable_lucas and not openai_key: missing.append("OpenAI")
if enable_sage and not gemini_key: missing.append("Google")
if enable_kai and not kimi_key: missing.append("Moonshot")
if enable_sage and not GEMINI_AVAILABLE:
    st.error("⚠️ Gemini ativo mas falta lib (`google-genai>=0.5` no requirements.txt)")

if go and question:
    if missing:
        st.error(f"⚠️ Faltam keys: {', '.join(set(missing))}")
    else:
        status_slot = st.empty()
        final_slot = st.empty()
        debate_slot = st.empty()
        ui_slots = {"status": status_slot, "final": final_slot, "debate": debate_slot}
        try:
            asyncio.run(run_council(question, ui_slots, agents))
        except Exception as e:
            st.error(f"Erro: {e}")
            st.exception(e)
elif go:
    st.warning("Digite uma pergunta primeiro.")
