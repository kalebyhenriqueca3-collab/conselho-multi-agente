"""
🧠 Conselho Multi-Agente v2 — 5 modelos com toggles
======================================================
Claude Opus · GPT-5 · Grok · Gemini 2.5 Pro · Kimi K2.6
"""

import streamlit as st
import asyncio
import os
import time
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="Conselho Multi-Agente", page_icon="🧠", layout="wide")

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
    footer { visibility: hidden; } #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuração")

    with st.expander("🔑 API Keys", expanded=not bool(os.getenv("ANTHROPIC_API_KEY"))):
        anthropic_key = st.text_input("Anthropic", type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""))
        openai_key = st.text_input("OpenAI", type="password",
            value=os.getenv("OPENAI_API_KEY", ""))
        xai_key = st.text_input("xAI (Grok)", type="password",
            value=os.getenv("XAI_API_KEY", ""))
        gemini_key = st.text_input("Google (Gemini)", type="password",
            value=os.getenv("GOOGLE_API_KEY", ""))
        kimi_key = st.text_input("Moonshot (Kimi)", type="password",
            value=os.getenv("MOONSHOT_API_KEY", ""))

    st.divider()
    st.subheader("🤖 Agentes ativos")
    enable_harper = st.checkbox("🔍 Harper · Grok (pesquisa)", value=True)
    enable_benjamin = st.checkbox("🧠 Benjamin · Claude Opus (lógica)", value=True)
    enable_lucas = st.checkbox("🎨 Lucas · GPT-5 (criatividade)", value=True)
    enable_sage = st.checkbox("📚 Sage · Gemini 2.5 Pro (contexto longo)", value=False)
    enable_kai = st.checkbox("⚡ Kai · Kimi K2.6 (execução técnica)", value=False)

    st.divider()
    st.subheader("🧠 Thinking / Reasoning")
    use_thinking = st.toggle("Claude extended thinking", value=True)
    thinking_budget = st.slider("Budget (tokens)", 1024, 16000, 5000, 1024,
                                disabled=not use_thinking)
    grok_reasoning = st.toggle("Grok reasoning mode", value=True)
    kimi_thinking = st.toggle("Kimi thinking mode", value=True)

    st.divider()
    st.subheader("🎯 Captain")
    captain_model = st.selectbox("Modelo do sintetizador",
        ["claude-opus-4-7", "claude-sonnet-4-6"], index=0)


def get_clients():
    clients = {
        "anthropic": AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None,
        "openai": AsyncOpenAI(api_key=openai_key) if openai_key else None,
        "grok": AsyncOpenAI(api_key=xai_key, base_url="https://api.x.ai/v1") if xai_key else None,
        "kimi": AsyncOpenAI(api_key=kimi_key, base_url="https://api.moonshot.ai/v1") if kimi_key else None,
        "gemini": None,
    }
    if gemini_key and GEMINI_AVAILABLE:
        clients["gemini"] = google_genai.Client(api_key=gemini_key)
    return clients


# ============================================================
# AGENTES
# ============================================================

async def agent_harper(prompt, clients):
    model = "grok-4.20-beta-0309-reasoning" if grok_reasoning else "grok-4-fast"
    resp = await clients["grok"].chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Você é Harper, especialista em pesquisa factual. "
                "Traga dados verificáveis, números, fontes, evidências. "
                "Responda em português brasileiro."
            )},
            {"role": "user", "content": prompt}
        ],
    )
    return {
        "text": resp.choices[0].message.content,
        "thinking": getattr(resp.choices[0].message, "reasoning_content", None)
    }


async def agent_benjamin(prompt, clients):
    kwargs = {
        "model": "claude-opus-4-7",
        "max_tokens": 4000,
        "system": (
            "Você é Benjamin, especialista em raciocínio lógico. "
            "Decomponha o problema, identifique premissas, aponte falhas. "
            "Responda em português brasileiro."
        ),
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["max_tokens"] = thinking_budget + 4000
    resp = await clients["anthropic"].messages.create(**kwargs)
    text, thinking = "", ""
    for block in resp.content:
        if block.type == "thinking": thinking = block.thinking
        elif block.type == "text": text = block.text
    return {"text": text, "thinking": thinking}


async def agent_lucas(prompt, clients):
    resp = await clients["openai"].chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": (
                "Você é Lucas, especialista em perspectivas criativas e UX. "
                "Traga ângulos não-óbvios, questione premissas. "
                "Responda em português brasileiro."
            )},
            {"role": "user", "content": prompt}
        ],
    )
    return {"text": resp.choices[0].message.content, "thinking": None}


async def agent_sage(prompt, clients):
    full_prompt = (
        "Você é Sage, especialista em síntese de informação ampla e "
        "conexões transversais. Faça leituras holísticas, integre "
        "perspectivas distintas, aponte padrões não-evidentes. "
        "Responda em português brasileiro.\n\n"
        f"Pergunta: {prompt}"
    )
    resp = await asyncio.to_thread(
        clients["gemini"].models.generate_content,
        model="gemini-2.5-pro",
        contents=full_prompt,
    )
    return {"text": resp.text, "thinking": None}


async def agent_kai(prompt, clients):
    extra = {} if kimi_thinking else {"thinking": {"type": "disabled"}}
    resp = await clients["kimi"].chat.completions.create(
        model="kimi-k2.6",
        messages=[
            {"role": "system", "content": (
                "Você é Kai, especialista em execução técnica concreta. "
                "Pense em implementação prática: ferramentas, passos, código, "
                "edge cases. Seja específico e acionável. "
                "Responda em português brasileiro."
            )},
            {"role": "user", "content": prompt}
        ],
        extra_body=extra,
        max_tokens=8000,
    )
    return {
        "text": resp.choices[0].message.content,
        "thinking": getattr(resp.choices[0].message, "reasoning_content", None)
    }


# ============================================================
# REGISTRY DINÂMICO
# ============================================================

ALL_AGENTS = [
    (lambda: enable_harper,    "Harper",   agent_harper,   "🔍", "Grok",            "#00d4ff"),
    (lambda: enable_benjamin,  "Benjamin", agent_benjamin, "🧠", "Claude Opus 4.7", "#ff6b9d"),
    (lambda: enable_lucas,     "Lucas",    agent_lucas,    "🎨", "GPT-5",           "#ffd93d"),
    (lambda: enable_sage,      "Sage",     agent_sage,     "📚", "Gemini 2.5 Pro",  "#9d4edd"),
    (lambda: enable_kai,       "Kai",      agent_kai,      "⚡", "Kimi K2.6",       "#06ffa5"),
]


def active_agents():
    return [(name, fn, emoji, label, color)
            for check, name, fn, emoji, label, color in ALL_AGENTS if check()]


# ============================================================
# CRÍTICA + CAPTAIN
# ============================================================

def build_critique_prompt(agent_name, question, all_outputs):
    others = "\n\n".join(
        f"### {n}:\n{o['text']}" for n, o in all_outputs.items() if n != agent_name
    )
    return (
        f"PERGUNTA ORIGINAL:\n{question}\n\n"
        f"RESPOSTAS DOS OUTROS:\n{others}\n\n"
        f"Você é {agent_name}. Identifique erros/contradições nos outros, "
        f"refine sua posição. Seja direto, sem diplomacia vazia."
    )


async def captain_synthesize(question, initial, critiques, clients, agents_meta):
    debate = ""
    for name, _, emoji, label, _ in agents_meta:
        debate += (
            f"\n\n=== {emoji} {name} ({label}) ===\n"
            f"[Inicial]\n{initial[name]['text']}\n\n"
            f"[Refinamento]\n{critiques[name]['text']}"
        )
    agent_list = ", ".join(f"{name} ({label})" for name, _, _, label, _ in agents_meta)

    kwargs = {
        "model": captain_model,
        "max_tokens": 4000,
        "system": (
            f"Você é o Captain. Integre as perspectivas dos agentes: {agent_list}. "
            "Resolva contradições explicitamente, descarte argumentos fracos, "
            "tome posição. Use formatação markdown clara. "
            "Responda em português brasileiro."
        ),
        "messages": [{
            "role": "user",
            "content": f"PERGUNTA:\n{question}\n\nDEBATE:{debate}\n\nProduza a resposta final."
        }],
    }
    if use_thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["max_tokens"] = thinking_budget + 4000
    resp = await clients["anthropic"].messages.create(**kwargs)
    text, thinking = "", ""
    for block in resp.content:
        if block.type == "thinking": thinking = block.thinking
        elif block.type == "text": text = block.text
    return {"text": text, "thinking": thinking}


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

async def run_council(question, ui_slots, agents):
    clients = get_clients()
    n = len(agents)

    ui_slots["status"].info(f"⚡ **Fase 1/3** — {n} agentes pensando em paralelo...")
    t0 = time.time()

    results_1 = await asyncio.gather(*[fn(question, clients) for _, fn, _, _, _ in agents])
    initial = {name: r for (name, _, _, _, _), r in zip(agents, results_1)}

    for (name, _, emoji, label, color), col in zip(agents, ui_slots["cols"]):
        with col:
            with st.expander(f"{emoji} **{name}** · {label}", expanded=True):
                if initial[name].get("thinking"):
                    with st.expander("💭 reasoning"):
                        st.caption(initial[name]["thinking"][:3000])
                st.markdown(initial[name]["text"])

    ui_slots["status"].info(f"⚡ **Fase 2/3** — Crítica cruzada paralela... ({time.time()-t0:.1f}s)")
    results_2 = await asyncio.gather(*[
        fn(build_critique_prompt(name, question, initial), clients)
        for name, fn, _, _, _ in agents
    ])
    critiques = {name: r for (name, _, _, _, _), r in zip(agents, results_2)}

    for (name, _, emoji, label, _), col in zip(agents, ui_slots["cols"]):
        with col:
            with st.expander(f"{emoji} {name} — refinamento"):
                st.markdown(critiques[name]["text"])

    ui_slots["status"].info(f"⚡ **Fase 3/3** — Captain sintetizando... ({time.time()-t0:.1f}s)")
    final = await captain_synthesize(question, initial, critiques, clients, agents)

    ui_slots["status"].success(f"✅ Concluído em {time.time()-t0:.1f}s · {n} agentes")

    with ui_slots["final"]:
        st.markdown('<div class="captain-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Resposta Final do Conselho")
        if final.get("thinking"):
            with st.expander("💭 thinking do Captain"):
                st.caption(final["thinking"][:5000])
        st.markdown(final["text"])
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# UI
# ============================================================

agents = active_agents()

st.markdown(f"""
<div class="main-header">
    <h1>🧠 Conselho Multi-Agente</h1>
    <p>{len(agents)} agentes ativos · pensamento paralelo + crítica cruzada + síntese</p>
</div>
""", unsafe_allow_html=True)

if len(agents) == 0:
    st.warning("⚠️ Ative pelo menos 1 agente na sidebar.")
elif len(agents) == 1:
    st.info("ℹ️ Com 1 agente só não tem debate. Ative pelo menos 2 pra ter crítica cruzada.")
else:
    cols_meta = st.columns(len(agents))
    for (name, _, emoji, label, color), col in zip(agents, cols_meta):
        with col:
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
                f"<b>{emoji} {name}</b><br><small style='opacity:0.7'>{label}</small></div>",
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

# Validação
missing = []
if not anthropic_key: missing.append("Anthropic (necessário pro Captain)")
if enable_harper and not xai_key: missing.append("xAI")
if enable_lucas and not openai_key: missing.append("OpenAI")
if enable_sage and not gemini_key: missing.append("Google")
if enable_kai and not kimi_key: missing.append("Moonshot")
if enable_sage and not GEMINI_AVAILABLE:
    st.error("⚠️ Gemini ativo mas falta a lib. Rode: `pip install google-genai`")

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
