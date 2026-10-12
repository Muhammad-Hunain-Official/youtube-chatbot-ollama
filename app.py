import os
import time
import random
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

# ---- Your backend (chatbot.py) ----
from Chatbot import extract_video_id, build_rag_chain, ask_question

# ---- Request limit per browser session ----
MAX_REQUESTS = 3

LIMIT_MESSAGE = (
    "😄 Hello dear! Lagta hai aap sirf check karne aaye hain ke is project ke "
    "peeche kya hai. Is ke requests limited hain, jo sab aap bhej dogay "
    "toh baki kya khaingay? 🍛 Aaj ke liye itna kafi hai, **bye bye!** 👋"
)

st.set_page_config(
    page_title="TubeChat AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
#  CACHING (so the same video never gets rebuilt)
# =====================================================================

@st.cache_resource(show_spinner=False)
def get_chain(video_id, transcript=None):
    return build_rag_chain(video_id, transcript)

# =====================================================================
#  UI DATA
# =====================================================================

THINK_WORDS = [
    "Thinking", "Pondering", "Analyzing", "Deciphering", "Synthesizing",
    "Contemplating", "Examining", "Unraveling", "Connecting the dots",
    "Distilling insights", "Cross-referencing", "Reflecting",
    "Decoding the transcript", "Piecing it together", "Brewing an answer",
    "Investigating", "Digesting the video", "Sifting through ideas",
]

QUICK_PROMPTS = [
    ("📝 Summarize", "Give me a clear summary of this video."),
    ("💡 Key takeaways", "What are the key takeaways from this video?"),
    ("🎯 Main topic", "What is the main topic of this video?"),
]

THEMES = {
    "night": {
        "bg1": "#0a0e1f", "bg2": "#1b1145", "bg3": "#0c2c4d",
        "text": "#eef1ff", "muted": "#a3acd6",
        "card": "rgba(255,255,255,0.06)",
        "border": "rgba(255,255,255,0.13)",
        "input": "rgba(255,255,255,0.08)",
        "input_solid": "#1a1f3d",
        "input_text": "#eef1ff",
        "a1": "#8b6cff", "a2": "#22d3ee",
        "shadow": "0 10px 40px rgba(0,0,0,0.45)",
        "orb": "0.45",
    },
    "day": {
        "bg1": "#fdf7ff", "bg2": "#e4efff", "bg3": "#ffeaf1",
        "text": "#1c1a3b", "muted": "#5d6189",
        "card": "rgba(255,255,255,0.72)",
        "border": "rgba(108,76,255,0.18)",
        "input": "rgba(255,255,255,0.95)",
        "input_solid": "#ffffff",
        "input_text": "#111111",
        "a1": "#6c4cff", "a2": "#0aa5d6",
        "shadow": "0 10px 35px rgba(108,76,255,0.15)",
        "orb": "0.35",
    },
}

STATIC_CSS = """
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}
.block-container {position: relative; z-index: 1; padding-top: 2rem; max-width: 1200px;}

/* ---------- animated background ---------- */
.stApp {
    background: linear-gradient(135deg, var(--bg1), var(--bg2), var(--bg3), var(--bg1));
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.bg-orb {
    position: fixed; border-radius: 50%; filter: blur(80px);
    opacity: var(--orb); z-index: -1; pointer-events: none;
}
.o1 {width: 380px; height: 380px; background: var(--a1); top: -90px; left: -90px; animation: float1 14s ease-in-out infinite;}
.o2 {width: 320px; height: 320px; background: var(--a2); bottom: -80px; right: -60px; animation: float2 17s ease-in-out infinite;}
.o3 {width: 220px; height: 220px; background: #ff6ec7; top: 45%; left: 55%; animation: float3 20s ease-in-out infinite;}
@keyframes float1 {0%,100%{transform: translate(0,0);} 50%{transform: translate(90px,70px);}}
@keyframes float2 {0%,100%{transform: translate(0,0);} 50%{transform: translate(-80px,-60px);}}
@keyframes float3 {0%,100%{transform: translate(0,0) scale(1);} 50%{transform: translate(-60px,40px) scale(1.25);}}

/* ---------- text ---------- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] {color: var(--text);}

/* ---------- hero ---------- */
.hero {text-align: center; padding: 1rem 0 1.6rem; animation: fadeUp .8s ease both;}
.hero .badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    background: var(--card); border: 1px solid var(--border);
    font-size: .8rem; letter-spacing: .5px; color: var(--muted);
    backdrop-filter: blur(10px); margin-bottom: 14px;
}
.hero h1 {font-size: 3.2rem; font-weight: 800; margin: 0; line-height: 1.15;}
.grad {
    background: linear-gradient(90deg, var(--a1), var(--a2), #ff6ec7, var(--a1));
    background-size: 300% 100%;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
}
@keyframes shine {0%{background-position: 0% 0;} 100%{background-position: 300% 0;}}
.hero p {color: var(--muted) !important; font-size: 1.1rem; margin-top: .6rem;}
@keyframes fadeUp {from{opacity: 0; transform: translateY(18px);} to{opacity: 1; transform: translateY(0);}}

/* ---------- glass cards ---------- */
.glass {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 1.3rem 1.4rem;
    backdrop-filter: blur(14px); box-shadow: var(--shadow);
    animation: fadeUp .7s ease both; height: 100%;
}
.glass h4 {margin: 0 0 .4rem; font-size: 1.05rem;}
.glass p {margin: 0; color: var(--muted) !important; font-size: .92rem;}
.glass .ico {font-size: 1.8rem; display: inline-block; animation: bob 3s ease-in-out infinite;}
@keyframes bob {0%,100%{transform: translateY(0);} 50%{transform: translateY(-6px);}}
.d2 {animation-delay: .15s;} .d3 {animation-delay: .3s;}

/* ---------- inputs ---------- */
.stTextInput [data-baseweb="input"],
.stTextInput [data-baseweb="base-input"],
.stTextArea [data-baseweb="textarea"],
.stTextArea [data-baseweb="base-input"] {
    background: var(--input-solid) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}
.stTextInput [data-baseweb="input"] > div,
.stTextArea [data-baseweb="textarea"] > div {
    background: transparent !important;
}
.stTextInput input,
.stTextArea textarea {
    background: transparent !important;
    color: var(--input-text) !important;
    -webkit-text-fill-color: var(--input-text) !important;
    caret-color: var(--a1) !important;
    border: none !important;
    padding: .8rem 1rem !important;
}
/* browser autofill can force its own colors */
.stTextInput input:-webkit-autofill {
    -webkit-text-fill-color: var(--input-text) !important;
    -webkit-box-shadow: 0 0 0 1000px var(--input-solid) inset !important;
}
.stTextInput [data-baseweb="input"]:focus-within {
    border-color: var(--a1) !important;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--a1) 30%, transparent) !important;
}
.stTextInput input::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    opacity: 1 !important;
}
/* ---------- buttons ---------- */
.stButton button {
    background: linear-gradient(135deg, var(--a1), var(--a2)) !important;
    color: #fff !important; border: none !important; border-radius: 14px !important;
    font-weight: 600; padding: .6rem 1.2rem; transition: all .25s ease;
    box-shadow: 0 6px 20px color-mix(in srgb, var(--a1) 40%, transparent);
}
.stButton button:hover {transform: translateY(-3px) scale(1.02); filter: brightness(1.1);}
.stButton button:active {transform: scale(.97);}
.stButton button p {color: #fff !important;}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 18px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    backdrop-filter: blur(12px); animation: fadeUp .5s ease both;
}
/* The bottom bar has its own solid background from Streamlit's base theme.
   Make it transparent, and give the input a SOLID color so the typed text
   is always readable. */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {background: transparent !important;}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: var(--input-solid) !important;
    border-radius: 18px !important;
}
[data-testid="stChatInput"] {
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow);
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--a2) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    opacity: 1 !important;
}
/* ---------- video ---------- */
[data-testid="stVideo"] {
    border-radius: 18px; overflow: hidden; box-shadow: var(--shadow);
    border: 1px solid var(--border); animation: fadeUp .7s ease both;
}

/* ---------- thinking animation ---------- */
.thinking {display: flex; align-items: center; gap: 12px; padding: 4px 0;}
.t-orb {
    width: 22px; height: 22px; border-radius: 50%;
    background: conic-gradient(var(--a1), var(--a2), #ff6ec7, var(--a1));
    animation: spin 1.1s linear infinite;
    box-shadow: 0 0 18px var(--a1);
}
@keyframes spin {to {transform: rotate(360deg);}}
.think-word {
    font-weight: 600; font-size: 1.05rem; letter-spacing: .3px;
    background: linear-gradient(90deg, var(--a1), var(--a2));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: wordIn .5s ease both;
}
@keyframes wordIn {from{opacity: 0; transform: translateY(8px);} to{opacity: 1; transform: translateY(0);}}
.dots i {
    display: inline-block; width: 6px; height: 6px; margin-left: 4px;
    border-radius: 50%; background: var(--a2); animation: blink 1.2s infinite;
}
.dots i:nth-child(2) {animation-delay: .2s;}
.dots i:nth-child(3) {animation-delay: .4s;}
@keyframes blink {0%,80%,100%{opacity: .2; transform: scale(.8);} 40%{opacity: 1; transform: scale(1.3);}}

/* ---------- empty state / footer ---------- */
.empty {text-align: center; color: var(--muted); padding: 2rem 0; animation: fadeUp .7s ease both;}
.empty .big {font-size: 2.6rem; display: block; animation: bob 3s ease-in-out infinite;}
.foot {text-align: center; color: var(--muted); font-size: .85rem; margin-top: 2.5rem;}
"""


# =====================================================================
#  UI HELPERS
# =====================================================================

def inject_theme(mode):
    t = THEMES[mode]
    variables = f""":root {{
        --bg1:{t['bg1']}; --bg2:{t['bg2']}; --bg3:{t['bg3']};
        --text:{t['text']}; --muted:{t['muted']};
        --card:{t['card']}; --border:{t['border']}; --input:{t['input']}; --input-solid:{t['input_solid']}; --input-text:{t['input_text']};
        --a1:{t['a1']}; --a2:{t['a2']}; --shadow:{t['shadow']}; --orb:{t['orb']};
    }}"""
    st.markdown(f"<style>{variables}{STATIC_CSS}</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="bg-orb o1"></div><div class="bg-orb o2"></div><div class="bg-orb o3"></div>',
        unsafe_allow_html=True,
    )


def thinking_html(word):
    return (
        '<div class="thinking"><div class="t-orb"></div>'
        f'<span class="think-word">{word}</span>'
        '<span class="dots"><i></i><i></i><i></i></span></div>'
    )


def answer_with_animation(chain, question, placeholder):
    """Runs the chain in a background thread while cycling thinking words."""
    words = THINK_WORDS[:]
    random.shuffle(words)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ask_question, chain, question)
        i = 0
        while not future.done():
            placeholder.markdown(
                thinking_html(words[i % len(words)]), unsafe_allow_html=True
            )
            i += 1
            time.sleep(1.1)
        return future.result()


def typewriter(placeholder, text):
    """Reveals the answer word by word."""
    tokens = text.split(" ")
    delay = 0.02 if len(tokens) < 250 else 0.008
    shown = ""
    for token in tokens:
        shown += token + " "
        placeholder.markdown(shown + "▌")
        time.sleep(delay)
    placeholder.markdown(text)


def toggle_theme():
    st.session_state.theme = "day" if st.session_state.theme == "night" else "night"


def set_pending(prompt):
    st.session_state.pending_q = prompt


# =====================================================================
#  SESSION STATE
# =====================================================================

st.session_state.setdefault("theme", "night")
st.session_state.setdefault("video_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("request_count", 0)
st.session_state.setdefault("pending_video_id", None)
st.session_state.setdefault("transcript_error", None)
st.session_state.setdefault("loaded_transcript", None)

inject_theme(st.session_state.theme)

AVATARS = {"user": "🧑", "assistant": "🤖"}

# =====================================================================
#  HEADER + THEME TOGGLE
# =====================================================================

_, toggle_col = st.columns([6, 1.3])
with toggle_col:
    label = "☀️ Day mode" if st.session_state.theme == "night" else "🌙 Night mode"
    st.button(label, on_click=toggle_theme, use_container_width=True)

st.markdown(
    """
    <div class="hero">
        <div class="badge">✨ LangChain · FAISS · Ollama</div>
        <h1>Chat with any <span class="grad">YouTube Video</span></h1>
        <p>Paste a link, and ask anything. Answers come straight from the video's transcript.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
#  VIDEO INPUT
# =====================================================================

in_col, btn_col = st.columns([5, 1.4])
with in_col:
    url_input = st.text_input(
        "YouTube link",
        placeholder="https://www.youtube.com/watch?v=...  (or just the 11-character video ID)",
        label_visibility="collapsed",
    )
with btn_col:
    load_clicked = st.button("🚀 Load video", use_container_width=True)


if load_clicked:
    vid = None

    try:
        vid = extract_video_id(url_input)

        with st.spinner(
            "Reading transcript and building knowledge base..."
        ):
            get_chain(vid, None)

        if vid != st.session_state.video_id:
            st.session_state.messages = []

        st.session_state.video_id = vid
        st.session_state.loaded_transcript = None
        st.session_state.pending_video_id = None
        st.session_state.transcript_error = None

        st.toast("Video ready! Ask away 🎉")

    except Exception as e:
        if vid:
            st.session_state.pending_video_id = vid
            st.session_state.transcript_error = str(e)
        else:
            st.error(f"Invalid YouTube URL: {e}")


if st.session_state.pending_video_id:
    st.warning(
        "Automatic transcript fetching failed. "
        "You can paste the transcript below to continue."
    )

    with st.expander(
        "Why did automatic loading fail?",
        expanded=True
    ):
        st.caption(st.session_state.transcript_error)

    with st.form("manual_transcript_form"):
        manual_transcript = st.text_area(
            "Paste video transcript here",
            height=220,
            placeholder=(
                "Open the video's transcript on YouTube, "
                "copy its text, and paste it here..."
            )
        )

        submit_transcript = st.form_submit_button(
            "Build chatbot from transcript",
            use_container_width=True
        )

    if submit_transcript:
        if not manual_transcript.strip():
            st.error("Please paste a transcript first.")

        else:
            vid = st.session_state.pending_video_id

            try:
                with st.spinner(
                    "Building knowledge base from transcript..."
                ):
                    get_chain(vid, manual_transcript.strip())

                if vid != st.session_state.video_id:
                    st.session_state.messages = []

                st.session_state.video_id = vid
                st.session_state.loaded_transcript = (
                    manual_transcript.strip()
                )
                st.session_state.pending_video_id = None
                st.session_state.transcript_error = None

                st.success(
                    "Transcript loaded successfully! "
                    "You can now ask questions."
                )

            except Exception as e:
                st.error(
                    f"Could not build chatbot: {e}"
                )
                
video_id = st.session_state.video_id

chain = (
    get_chain(
        video_id,
        st.session_state.loaded_transcript
    )
    if video_id
    else None
)
# Quick-prompt click (if any), otherwise the chat box
pending = st.session_state.pop("pending_q", None)
typed = st.chat_input(
    "Ask something about the video..." if chain else "Load a video first...",
    disabled=chain is None,
)
question = typed or pending

# =====================================================================
#  MAIN AREA
# =====================================================================

if not chain:
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="glass"><span class="ico">🔗</span><h4>1. Paste a link</h4>'
        "<p>Any YouTube URL, Shorts link, or plain video ID works.</p></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="glass d2"><span class="ico">🧠</span><h4>2. We read the video</h4>'
        "<p>The transcript is split, embedded and stored in a fast FAISS index.</p></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        '<div class="glass d3"><span class="ico">💬</span><h4>3. Ask anything</h4>'
        "<p>Get clear, transcript-grounded answers in seconds.</p></div>",
        unsafe_allow_html=True,
    )
else:
    video_col, chat_col = st.columns([1, 1.4], gap="large")

    with video_col:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Quick questions")
        for idx, (btn_label, prompt_text) in enumerate(QUICK_PROMPTS):
            st.button(
                btn_label,
                key=f"qp_{idx}",
                on_click=set_pending,
                args=(prompt_text,),
                use_container_width=True,
            )

    with chat_col:
        if not st.session_state.messages and not question:
            st.markdown(
                '<div class="empty"><span class="big">🎬</span>'
                "Your video is ready.<br>Ask a question below or tap a quick prompt.</div>",
                unsafe_allow_html=True,
            )

        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
                st.markdown(m["content"])

        if question:
            with st.chat_message("user", avatar=AVATARS["user"]):
                st.markdown(question)

            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                if st.session_state.request_count >= MAX_REQUESTS:
                    # Limit reached: show the funny message, skip the LLM call
                    answer = LIMIT_MESSAGE
                    st.markdown(answer)
                else:
                    st.session_state.request_count += 1
                    box = st.empty()
                    try:
                        answer = answer_with_animation(chain, question, box)
                        typewriter(box, answer)
                    except Exception as e:
                        answer = f"⚠️ Something went wrong: {e}"
                        box.markdown(answer)

            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": answer})
st.markdown(
    '<div class="foot">Built with LangChain · Ollama · '
    'FAISS · Streamlit 💜</div>',
    unsafe_allow_html=True,
)