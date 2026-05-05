"""
🧠 Conselho Multi-Agente v4 — controle total
================================================
Você escolhe pra cada agente:
  - Se está ativo
  - Qual modelo específico
  - Qual nível de thinking/effort
E vê o custo estimado ANTES de gastar.
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

st.set_page_config(page_title="Conselho Multi-Agente v4", page_icon="🧠", layout="wide")

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
        background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
        border: 2px solid rgba(102,126,234,0.4);
        padding: 2rem; border-radius: 16px; margin-top: 1rem;
        box-shadow: 0 10px 40px rgba(102,126,234,0.2);
    }
    .cost-card {
        padding: 1rem; border-radius: 12px; margin: 1rem 0;
        background: rgba(255,193,7,0.1); border: 1px solid rgba(255,193,7,0.3);
    }
    footer { visibility: hidden; } #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CATÁLOGO DE MODELOS / PRICING (USD por 1M tokens, aprox.)
# ============================================================

CLAUDE_MODELS = {
    "claude-opus-4-7":  {"label": "Claude Opus 4.7 (top)",     "in":  5.0, "out": 25.0},
    "claude-opus-4-6":  {"label": "Claude Opus 4.6",            "in":  5.0, "out": 25.0},
    "claude-sonnet-4-6":{"label": "Claude Sonnet 4.6 (barato)", "in":  3.0, "out": 15.0},
}

GPT_MODELS = {
    "gpt-5.5":    {"label": "GPT-5.5 (top)",     "in": 1.75, "out": 14.0},
    "gpt-5.4":    {"label": "GPT-5.4",           "in": 1.50, "out": 12.0},
    "gpt-5.2":    {"label": "GPT-5.2",           "in": 1.50, "out": 12.0},
    "gpt-4o":     {"label": "GPT-4o (sem thinking)", "in": 2.50, "out": 10.0},
}

GROK_MODELS = {
    "grok-4.20-multi-agent": {"label": "Grok 4.2 Multi-Agent (top)", "in": 2.0, "out": 6.0},
    "grok-4.20-beta-0309-reasoning": {"label": "Grok 4.2 Reasoning",  "in": 1.5, "out": 5.0},
    "grok-4-fast":           {"label": "Grok 4 Fast (barato)",        "in": 0.5, "out": 2.0},
}

GEMINI_MODELS = {
    "gemini-3.1-pro-preview": {"label": "Gemini 3.1 Pro (top, Deep Think Mini)", "in": 2.0, "out": 12.0},
    "gemini-3-pro":           {"label": "Gemini 3 Pro",                          "in": 1.5, "out": 10.0},
    "gemini-2.5-pro":         {"label": "Gemini 2.5 Pro",                        "in": 1.25, "out": 10.0},
}

KIMI_MODELS = {
    "kimi-k2.6": {"label": "Kimi K2.6 (top)", "in": 0.74, "out": 3.49},
    "kimi-k2.5": {"label": "Kimi K2.5",       "in": 0.50, "out": 2.50},
}

# Multiplicador de tokens estimados por effort (impacta output tokens)
EFFORT_MULTIPLIERS = {
    "none":   0.5,
    "low":    1.0,
    "medium": 2.0,
    "high":   4.0,
    "xhigh":  7.0,
    "max":   10.0,
}


# ============================================================
# SIDEBAR — controle total
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuração")

    # ─── Presets rápidos ───
    st.subheader("⚡ Presets rápidos")
    preset_cols = st.columns(3)
    if preset_cols[0].button("🪙 Econômico", use_container_width=True, help="Modelos baratos, effort low"):
        st.session_state["preset"] = "economy"
    if preset_cols[1].button("⚖️ Balanceado", use_container_width=True, help="Modelos top, effort medium"):
        st.session_state["preset"] = "balanced"
    if preset_cols[2].button("🚀 Top tier", use_container_width=True, help="Tudo no MAX"):
        st.session_state["preset"] = "top"

    preset = st.session_state.get("preset", None)

    # Define defaults por preset
    def preset_val(economy, balanced, top, default):
        if preset == "economy":  return economy
        if preset == "balanced": return balanced
        if preset == "top":      return top
        return default

    st.divider()

    # ─── API Keys ───
    with st.expander("🔑 API Keys", expanded=not bool(os.getenv("ANTHROPIC_API_KEY"))):
        anthropic_key = st.text_input("Anthropic", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        openai_key = st.text_input("OpenAI", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        xai_key = st.text_input("xAI (Grok)", type="password", value=os.getenv("XAI_API_KEY", ""))
        gemini_key = st.text_input("Google (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        kimi_key = st.text_input("Moonshot (Kimi)", type="password", value=os.getenv("MOONSHOT_API_KEY", ""))

    st.divider()
    st.subheader("🤖 Agentes")

    # ─── Harper (Grok) ───
    with st.expander("🔍 Harper · Grok", expanded=True):
        enable_harper = st.checkbox("Ativar", value=True, key="en_harper")
        harper_model = st.selectbox(
            "Modelo",
            list(GROK_MODELS.keys()),
            index=preset_val(2, 0, 0, 0),
            format_func=lambda k: GROK_MODELS[k]["label"],
            key="m_harper"
        )
        # Grok não tem effort selectable — o reasoning model já vem com seu próprio thinking
        st.caption("ℹ️ Grok reasoning é embutido no modelo")

    # ─── Benjamin (Claude principal) ───
    with st.expander("🧠 Benjamin · Claude (lógica)", expanded=True):
        enable_benjamin = st.checkbox("Ativar", value=True, key="en_benjamin")
        benjamin_model = st.selectbox(
            "Modelo",
            list(CLAUDE_MODELS.keys()),
            index=preset_val(2, 0, 0, 0),
            format_func=lambda k: CLAUDE_MODELS[k]["label"],
            key="m_benjamin"
        )
        benjamin_effort = st.select_slider(
            "Effort",
            options=["low", "medium", "high", "xhigh", "max"],
            value=preset_val("low", "medium", "max", "high"),
            key="e_benjamin"
        )

    # ─── Aria (Claude alt) ───
    with st.expander("⚖️ Aria · Claude (nuance)"):
        enable_aria = st.checkbox("Ativar", value=False, key="en_aria")
        aria_model = st.selectbox(
            "Modelo",
            list(CLAUDE_MODELS.keys()),
            index=1,
            format_func=lambda k: CLAUDE_MODELS[k]["label"],
            key="m_aria"
        )
        aria_effort = st.select_slider(
            "Effort",
            options=["low", "medium", "high", "xhigh", "max"],
            value=preset_val("low", "medium", "max", "high"),
            key="e_aria"
        )

    # ─── Lucas (GPT) ───
    with st.expander("🎨 Lucas · GPT (criatividade)", expanded=True):
        enable_lucas = st.checkbox("Ativar", value=True, key="en_lucas")
        lucas_model = st.selectbox(
            "Modelo",
            list(GPT_MODELS.keys()),
            index=preset_val(3, 0, 0, 0),
            format_func=lambda k: GPT_MODELS[k]["label"],
            key="m_lucas"
        )
        if lucas_model != "gpt-4o":
            lucas_effort = st.select_slider(
                "Reasoning effort",
                options=["none", "low", "medium", "high", "xhigh"],
                value=preset_val("low", "medium", "xhigh", "high"),
                key="e_lucas"
            )
        else:
            lucas_effort = "none"
            st.caption("ℹ️ GPT-4o não tem reasoning")

    # ─── Sage (Gemini) ───
    with st.expander("📚 Sage · Gemini (contexto longo)"):
        enable_sage = st.checkbox("Ativar", value=False, key="en_sage")
        sage_model = st.selectbox(
            "Modelo",
            list(GEMINI_MODELS.keys()),
            index=0,
            format_func=lambda k: GEMINI_MODELS[k]["label"],
            key="m_sage"
        )
        sage_thinking = st.select_slider(
            "Thinking level",
            options=["low", "medium", "high"],
            value=preset_val("low", "medium", "high", "medium"),
            key="t_sage"
        )

    # ─── Kai (Kimi) ───
    with st.expander("⚡ Kai · Kimi (execução técnica)"):
        enable_kai = st.checkbox("Ativar", value=False, key="en_kai")
        kai_model = st.selectbox(
            "Modelo",
            list(KIMI_MODELS.keys()),
            index=0,
            format_func=lambda k: KIMI_MODELS[k]["label"],
            key="m_kai"
        )
        kai_thinking = st.toggle("Thinking ON", value=True, key="t_kai")

    st.divider()
    st.subheader("🎯 Captain (sintetizador)")
    captain_model = st.selectbox(
        "Modelo",
        list(CLAUDE_MODELS.keys()),
        index=preset_val(2, 1, 0, 1),
        format_func=lambda k: CLAUDE_MODELS[k]["label"],
        key="m_captain"
    )
    captain_effort = st.select_slider(
        "Effort",
        options=["low", "medium", "high", "xhigh", "max"],
        value=preset_val("low", "medium", "max", "high"),
        key="e_captain"
    )


# ============================================================
# CLIENTS
# ============================================================
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


# ============================================================
# AGENTES
# ============================================================

async def agent_harper(prompt, clients):
    resp = await clients["grok"].chat.completions.create(
        model=harper_model,
        messages=[
            {"role": "system", "content": (
                "Você é Harper, especialista em pesquisa factual. "
                "Traga dados verificáveis, números, fontes. Responda em pt-BR."
            )},
            {"role": "user", "content": prompt}
        ],
    )
    return {
        "text": resp.choices[0].message.content,
        "thinking": getattr(resp.choices[0].message, "reasoning_content", None)
    }


async def _claude_call(model, effort, system, prompt, client):
    """Claude com adaptive thinking + effort configurável."""
    kwargs = {
        "model": model,
        "max_tokens": 64000 if effort in ("xhigh", "max") else 32000,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model == "claude-opus-4-7":
        # 4.7 só aceita adaptive
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        kwargs["output_config"] = {"effort": effort}
    elif model in ("claude-opus-4-6", "claude-sonnet-4-6"):
        # 4.6 aceita ambos — usamos adaptive (recomendado)
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}

    resp = await client.messages.create(**kwargs)
    text, thinking = "", ""
    for block in resp.content:
        if block.type == "thinking":
            thinking = block.thinking or ""
        elif block.type == "text":
            text = block.text
    return {"text": text, "thinking": thinking}


async def agent_benjamin(prompt, clients):
    return await _claude_call(
        benjamin_model, benjamin_effort,
        ("Você é Benjamin, especialista em raciocínio lógico estrutural. "
         "Decomponha em premissas, identifique falhas, construa argumentos rigorosos. "
         "Responda em pt-BR."),
        prompt, clients["anthropic"]
    )


async def agent_aria(prompt, clients):
    return await _claude_call(
        aria_model, aria_effort,
        ("Você é Aria, especialista em nuance e tradeoffs. "
         "Pondere prós e contras, identifique tensões entre objetivos, considerações "
         "éticas e de longo prazo. Responda em pt-BR."),
        prompt, clients["anthropic"]
    )


async def agent_lucas(prompt, clients):
    """GPT-5.x via Responses API com effort. GPT-4o via Chat Completions."""
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
    text = getattr(resp, "output_text", "") or ""
    if not text and hasattr(resp, "output"):
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", None) == "output_text":
                        text += c.text
    return {"text": text, "thinking": None}


async def agent_sage(prompt, clients):
    full_prompt = (
        "Você é Sage, síntese ampla e conexões transversais. "
        "Leituras holísticas, padrões não-evidentes. Responda em pt-BR.\n\n"
        f"Pergunta: {prompt}"
    )
    config = gemini_types.GenerateContentConfig(
        thinking_config=gemini_types.ThinkingConfig(thinking_level=sage_thinking),
    ) if sage_model.startswith("gemini-3") else None

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
    (lambda: enable_harper,   "Harper",   agent_harper,   "🔍", lambda: GROK_MODELS[harper_model]["label"],     "#00d4ff", lambda: GROK_MODELS[harper_model],    None),
    (lambda: enable_benjamin, "Benjamin", agent_benjamin, "🧠", lambda: f"{CLAUDE_MODELS[benjamin_model]['label']} · {benjamin_effort}", "#ff6b9d", lambda: CLAUDE_MODELS[benjamin_model], lambda: benjamin_effort),
    (lambda: enable_aria,     "Aria",     agent_aria,     "⚖️", lambda: f"{CLAUDE_MODELS[aria_model]['label']} · {aria_effort}",         "#ff9d6b", lambda: CLAUDE_MODELS[aria_model],     lambda: aria_effort),
    (lambda: enable_lucas,    "Lucas",    agent_lucas,    "🎨", lambda: f"{GPT_MODELS[lucas_model]['label']} · {lucas_effort}",          "#ffd93d", lambda: GPT_MODELS[lucas_model],       lambda: lucas_effort),
    (lambda: enable_sage,     "Sage",     agent_sage,     "📚", lambda: f"{GEMINI_MODELS[sage_model]['label']} · {sage_thinking}",       "#9d4edd", lambda: GEMINI_MODELS[sage_model],     lambda: sage_thinking),
    (lambda: enable_kai,      "Kai",      agent_kai,      "⚡", lambda: KIMI_MODELS[kai_model]["label"],         "#06ffa5", lambda: KIMI_MODELS[kai_model],        None),
]


def active_agents():
    return [(name, fn, emoji, label_fn, color, pricing_fn, effort_fn)
            for check, name, fn, emoji, label_fn, color, pricing_fn, effort_fn in ALL_AGENTS
            if check()]


# ============================================================
# ESTIMADOR DE CUSTO
# ============================================================

def estimate_cost(agents, captain_pricing, captain_eff):
    """Estima custo. Assume ~2k input tokens e output baseado no effort."""
    INPUT_TOKENS = 2000   # pergunta + system prompt
    INPUT_DEBATE = 8000   # crítica vê todos os outputs

    total = 0.0
    breakdown = []

    # Cada agente: 2 chamadas (inicial + crítica)
    for name, _, emoji, label_fn, _, pricing_fn, effort_fn in agents:
        pricing = pricing_fn()
        effort = effort_fn() if effort_fn else "medium"
        mult = EFFORT_MULTIPLIERS.get(effort, 2.0)
        out_tokens_per_call = 1500 * mult  # output estimate
        # Chamada 1 (inicial)
        cost_1 = (INPUT_TOKENS * pricing["in"] + out_tokens_per_call * pricing["out"]) / 1_000_000
        # Chamada 2 (crítica)
        cost_2 = (INPUT_DEBATE * pricing["in"] + out_tokens_per_call * pricing["out"]) / 1_000_000
        agent_total = cost_1 + cost_2
        total += agent_total
        breakdown.append((f"{emoji} {name}", agent_total))

    # Captain
    captain_input = INPUT_DEBATE * 2  # vê inicial + crítica de todos
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
        f"Você é {agent_name}. Identifique erros, contradições ou pontos fracos "
        f"nas respostas dos outros. Refine sua posição. Seja direto."
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
            f"Você é o Captain. Integre as perspectivas dos agentes ({agent_list}). "
            "Resolva contradições, descarte argumentos fracos, tome posição clara. "
            "Use markdown com seções. Responda em pt-BR."
        ),
        "messages": [{
            "role": "user",
            "content": f"PERGUNTA:\n{question}\n\nDEBATE:{debate}\n\nProduza a resposta final."
        }],
    }
    if captain_model == "claude-opus-4-7":
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        kwargs["output_config"] = {"effort": captain_effort}
    else:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": captain_effort}

    resp = await clients["anthropic"].messages.create(**kwargs)
    text, thinking = "", ""
    for block in resp.content:
        if block.type == "thinking":
            thinking = block.thinking or ""
        elif block.type == "text":
            text = block.text
    return {"text": text, "thinking": thinking}


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

async def run_council(question, ui_slots, agents):
    clients = get_clients()
    n = len(agents)

    ui_slots["status"].info(f"⚡ **Fase 1/3** — {n} agentes em paralelo...")
    t0 = time.time()

    results_1 = await asyncio.gather(
        *[fn(question, clients) for _, fn, _, _, _, _, _ in agents],
        return_exceptions=True
    )
    initial = {}
    for (name, _, _, _, _, _, _), r in zip(agents, results_1):
        initial[name] = r if not isinstance(r, Exception) else {"text": f"⚠️ {r}", "thinking": None}

    for (name, _, emoji, label_fn, _, _, _), col in zip(agents, ui_slots["cols"]):
        with col:
            with st.expander(f"{emoji} **{name}** · {label_fn()}", expanded=True):
                if initial[name].get("thinking"):
                    with st.expander("💭 thinking"):
                        st.caption(initial[name]["thinking"][:4000])
                st.markdown(initial[name]["text"])

    ui_slots["status"].info(f"⚡ **Fase 2/3** — Crítica cruzada... ({time.time()-t0:.1f}s)")

    results_2 = await asyncio.gather(
        *[fn(build_critique_prompt(name, question, initial), clients)
          for name, fn, _, _, _, _, _ in agents],
        return_exceptions=True
    )
    critiques = {}
    for (name, _, _, _, _, _, _), r in zip(agents, results_2):
        critiques[name] = r if not isinstance(r, Exception) else {"text": f"⚠️ {r}", "thinking": None}

    for (name, _, emoji, _, _, _, _), col in zip(agents, ui_slots["cols"]):
        with col:
            with st.expander(f"{emoji} {name} — refinamento"):
                st.markdown(critiques[name]["text"])

    ui_slots["status"].info(f"⚡ **Fase 3/3** — Captain ({captain_effort})... ({time.time()-t0:.1f}s)")
    final = await captain_synthesize(question, initial, critiques, clients, agents)

    ui_slots["status"].success(f"✅ {time.time()-t0:.1f}s · {n} agentes")

    with ui_slots["final"]:
        st.markdown('<div class="captain-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Resposta Final do Conselho")
        if final.get("thinking"):
            with st.expander("💭 thinking do Captain"):
                st.caption(final["thinking"][:8000])
        st.markdown(final["text"])
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# UI
# ============================================================

agents = active_agents()

st.markdown(f"""
<div class="main-header">
    <h1>🧠 Conselho Multi-Agente v4</h1>
    <p>{len(agents)} agentes ativos · controle total · estimador de custo</p>
</div>
""", unsafe_allow_html=True)

# Estimador de custo
if len(agents) >= 2:
    cost, breakdown = estimate_cost(agents, CLAUDE_MODELS[captain_model], captain_effort)
    cost_low = cost * 0.4
    cost_high = cost * 2.5

    color = "#22c55e" if cost < 0.5 else "#f59e0b" if cost < 2.0 else "#ef4444"

    st.markdown(f"""
    <div class="cost-card" style="border-color: {color}; background: {color}15;">
        <b style="color:{color}">💰 Custo estimado: ${cost_low:.2f} – ${cost_high:.2f} por pergunta</b><br>
        <small style="opacity:0.8">Estimativa central: ${cost:.2f} · varia conforme tamanho da pergunta e thinking real</small>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Ver breakdown por agente"):
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
question = st.text_area(
    "Sua pergunta:",
    placeholder="Ex: Qual a melhor arquitetura pra integrar HubSpot, Salesforce e NetSuite?",
    height=100,
)

col_a, _ = st.columns([1, 4])
with col_a:
    go = st.button("🚀 Convocar conselho", type="primary",
                   use_container_width=True, disabled=len(agents) < 2)

# Validação keys
missing = []
if not anthropic_key: missing.append("Anthropic")
if enable_harper and not xai_key: missing.append("xAI")
if enable_lucas and not openai_key: missing.append("OpenAI")
if enable_sage and not gemini_key: missing.append("Google")
if enable_kai and not kimi_key: missing.append("Moonshot")
if enable_sage and not GEMINI_AVAILABLE:
    st.error("⚠️ Gemini ativo mas falta lib. Adiciona `google-genai>=0.5` no requirements.txt")

if go and question:
    if missing:
        st.error(f"⚠️ Faltam keys: {', '.join(set(missing))}")
    else:
        status_slot = st.empty()
        cols = st.columns(len(agents))
        final_slot = st.empty()
        ui_slots = {"status": status_slot, "cols": cols, "final": final_slot}
        try:
            asyncio.run(run_council(question, ui_slots, agents))
        except Exception as e:
            st.error(f"Erro: {e}")
            st.exception(e)
elif go:
    st.warning("Digite uma pergunta primeiro.")
