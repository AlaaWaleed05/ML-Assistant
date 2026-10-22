import hashlib
import hmac
import os
import re
import secrets
import sqlite3

import streamlit as st
from openai import (
    OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError,
)

# ───────────────────────── Config ─────────────────────────
st.set_page_config(page_title="ML Buddy", page_icon="🧠", layout="wide")

DB_PATH = "users.db"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


API_KEY = _secret("GEMINI_API_KEY")
MODEL = _secret("MODEL", "gemini-2.5-flash")


@st.cache_resource
def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ───────────────────────── Database ─────────────────────────
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                salt TEXT NOT NULL,
                pw_hash TEXT NOT NULL,
                preferred_name TEXT
            )"""
        )


def hash_pw(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 200_000
    ).hex()


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def sign_up(name, email, password):
    name, email = name.strip(), email.strip().lower()
    if not name or not email or not password:
        return None, "Please fill in all fields."
    if not EMAIL_RE.match(email):
        return None, "That email doesn't look right."
    if len(password) < 6:
        return None, "Password must be at least 6 characters."
    salt = secrets.token_hex(16)
    try:
        with db() as c:
            c.execute(
                "INSERT INTO users (name, email, salt, pw_hash) VALUES (?,?,?,?)",
                (name, email, salt, hash_pw(password, salt)),
            )
    except sqlite3.IntegrityError:
        return None, "This email is already registered. Try logging in."
    return get_user_by_email(email), None


def log_in(email, password):
    email = email.strip().lower()
    row = get_user_by_email(email)
    if not row or not hmac.compare_digest(
        row["pw_hash"], hash_pw(password, row["salt"])
    ):
        return None, "Wrong email or password."
    return row, None


def get_user_by_email(email):
    with db() as c:
        return c.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def set_preferred_name(user_id, new_name):
    with db() as c:
        c.execute("UPDATE users SET preferred_name = ? WHERE id = ?", (new_name, user_id))


def display_name(user):
    return user["preferred_name"] or user["name"]


# ───────────────────────── LLM ─────────────────────────
BASE_INSTRUCTIONS = """
You are ML Buddy, an expert and friendly Machine Learning teacher and assistant.

SCOPE:
- Answer only questions related to Machine Learning (definitions, types of ML: supervised,
  unsupervised, semi-supervised, reinforcement; algorithms; training/evaluation; overfitting;
  metrics; deep learning basics; data preprocessing; etc.).
- If the question is not about Machine Learning, politely say you only help with ML topics
  (in the same language as the question).

STYLE:
- Default style: simple, clear, and well organized, with short examples when helpful.
- If the user asks for a specific style (e.g. "explain like I'm a beginner", "with examples",
  "in detail", "short"), follow it exactly.

LANGUAGE (very important):
- Reply in the same language/dialect the user wrote in.
- English question -> answer in English.
- Egyptian colloquial Arabic question -> answer in Egyptian colloquial Arabic.
- Modern Standard Arabic (fusha) question -> answer in Modern Standard Arabic.
- Keep technical terms (e.g. overfitting, gradient descent) in English when it helps clarity.

FORMAT:
- You may use simple Markdown (short headings, bold, bullet lists, code blocks).
- Do not use LaTeX or $$ formulas; write formulas in plain text or code blocks.

{first_turn_rule}
NAME:
- The user's name is: {name}. Use it naturally now and then, not in every message.
- If the user explicitly asks to be called by a different name, accept happily and use the new
  name from now on. In that case, add this exact tag on a NEW LAST LINE of your reply and
  nothing after it: [[SET_NAME: <new name>]]
- Never output that tag in any other situation.
"""

NAME_TAG = re.compile(r"\[\[SET_NAME:\s*(.+?)\s*\]\]", re.S)


def friendly_error(e):
    if isinstance(e, AuthenticationError):
        return "Authentication error: check your API key."
    if isinstance(e, RateLimitError):
        return "Rate limit or quota exceeded. Please try again in a moment."
    if isinstance(e, APIConnectionError):
        return "Connection error: check your internet connection."
    if isinstance(e, APIError):
        return f"API error: {e}"
    return f"Unexpected error: {e}"


FIRST_TURN_RULE = """
FIRST MESSAGE OF THE SESSION:
- Start your reply with a short greeting that uses the user's name, in the language of their message:
  English -> "Hello {name}!"  |  Egyptian Arabic -> "أهلاً بيك يا {name}!"  |  Fusha -> "أهلاً بك يا {name}!"
- Then answer the question right away on a new line.
"""


def stream_reply(history, name, raw_holder, first_turn=False):
    """Yield visible text; hide the trailing [[SET_NAME: ...]] tag; store raw text."""
    rule = FIRST_TURN_RULE.format(name=name) if first_turn else ""
    messages = [{"role": "system", "content": BASE_INSTRUCTIONS.format(name=name, first_turn_rule=rule)}]
    messages += history[-20:]
    stream = get_client().chat.completions.create(
        model=MODEL, messages=messages, temperature=0.5, max_tokens=2000, stream=True
    )
    full, emitted = "", 0
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content or ""
        if not piece:
            continue
        full += piece
        idx = full.find("[[")
        safe_end = idx if idx != -1 else max(len(full) - 1, emitted)
        if safe_end > emitted:
            yield full[emitted:safe_end]
            emitted = safe_end
    idx = full.find("[[")
    end = idx if idx != -1 else len(full)
    if end > emitted:
        yield full[emitted:end].rstrip()
    raw_holder.append(full)


# ───────────────────────── Styling ─────────────────────────
NEURAL_SVG = """
<svg viewBox="0 0 520 420" xmlns="http://www.w3.org/2000/svg" class="neural">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#7C3AED"/><stop offset="1" stop-color="#06B6D4"/>
    </linearGradient>
  </defs>
  <g stroke="url(#g)" stroke-width="1.6" opacity=".45" fill="none">
    <line x1="70" y1="90" x2="230" y2="60"/><line x1="70" y1="90" x2="230" y2="170"/>
    <line x1="70" y1="90" x2="230" y2="280"/><line x1="70" y1="210" x2="230" y2="60"/>
    <line x1="70" y1="210" x2="230" y2="170"/><line x1="70" y1="210" x2="230" y2="280"/>
    <line x1="70" y1="330" x2="230" y2="170"/><line x1="70" y1="330" x2="230" y2="280"/>
    <line x1="70" y1="330" x2="230" y2="360"/><line x1="70" y1="210" x2="230" y2="360"/>
    <line x1="230" y1="60" x2="390" y2="120"/><line x1="230" y1="170" x2="390" y2="120"/>
    <line x1="230" y1="170" x2="390" y2="260"/><line x1="230" y1="280" x2="390" y2="120"/>
    <line x1="230" y1="280" x2="390" y2="260"/><line x1="230" y1="360" x2="390" y2="260"/>
    <line x1="390" y1="120" x2="480" y2="190"/><line x1="390" y1="260" x2="480" y2="190"/>
  </g>
  <g>
    <circle class="n n1" cx="70" cy="90" r="14" fill="#7C3AED"/>
    <circle class="n n2" cx="70" cy="210" r="14" fill="#EC4899"/>
    <circle class="n n3" cx="70" cy="330" r="14" fill="#06B6D4"/>
    <circle class="n n2" cx="230" cy="60" r="14" fill="#F59E0B"/>
    <circle class="n n3" cx="230" cy="170" r="14" fill="#7C3AED"/>
    <circle class="n n1" cx="230" cy="280" r="14" fill="#EC4899"/>
    <circle class="n n2" cx="230" cy="360" r="14" fill="#06B6D4"/>
    <circle class="n n1" cx="390" cy="120" r="14" fill="#06B6D4"/>
    <circle class="n n3" cx="390" cy="260" r="14" fill="#F59E0B"/>
    <circle class="n n2" cx="480" cy="190" r="16" fill="#7C3AED"/>
  </g>
</svg>
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Cairo:wght@400;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Poppins','Cairo',sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 8% 10%, rgba(124,58,237,.14), transparent 38%),
    radial-gradient(circle at 92% 18%, rgba(6,182,212,.16), transparent 40%),
    radial-gradient(circle at 80% 90%, rgba(236,72,153,.13), transparent 40%),
    #ffffff;
}
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1150px; }

.brand { font-weight: 800; font-size: 1.1rem; color: #4C1D95; letter-spacing:.3px; }
.hero-title {
  font-size: 3.1rem; line-height: 1.08; font-weight: 800; margin: .4rem 0 .8rem;
  background: linear-gradient(90deg,#7C3AED,#EC4899 55%,#F59E0B);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { font-size: 1.08rem; color: #475569; margin-bottom: 1.2rem; }
.chips span {
  display:inline-block; padding:.35rem .8rem; margin:0 .4rem .5rem 0; border-radius:999px;
  font-size:.82rem; font-weight:600; color:#4C1D95; background:#EDE9FE; border:1px solid #DDD6FE;
}
.chips span:nth-child(2){ background:#CFFAFE; color:#155E75; border-color:#A5F3FC; }
.chips span:nth-child(3){ background:#FCE7F3; color:#9D174D; border-color:#FBCFE8; }
.chips span:nth-child(4){ background:#FEF3C7; color:#92400E; border-color:#FDE68A; }

.neural { width: 100%; max-width: 440px; margin-top: .5rem; }
.neural .n { animation: pulse 2.6s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
.neural .n2 { animation-delay: .8s; } .neural .n3 { animation-delay: 1.6s; }
@keyframes pulse { 0%,100%{ transform: scale(1); } 50%{ transform: scale(1.35); } }

/* auth card */
div[data-testid="stTabs"] {
  background: rgba(255,255,255,.85); backdrop-filter: blur(8px);
  border: 1px solid #E9D5FF; border-radius: 22px; padding: 1.2rem 1.4rem 1.4rem;
  box-shadow: 0 18px 50px rgba(124,58,237,.14);
}
.stTextInput input { border-radius: 12px; }
.stButton > button, .stFormSubmitButton > button {
  border-radius: 12px; font-weight: 700; border: none; color: white;
  background: linear-gradient(90deg,#7C3AED,#EC4899); transition: transform .15s, box-shadow .15s;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-2px); box-shadow: 0 10px 24px rgba(236,72,153,.35); color: white;
}

/* chat */
[data-testid="stChatMessage"] {
  border-radius: 18px; padding: .9rem 1.1rem; margin-bottom: .6rem;
  background: rgba(255,255,255,.9); border: 1px solid #EDE9FE;
  box-shadow: 0 4px 16px rgba(124,58,237,.06);
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
  unicode-bidi: plaintext; text-align: start;
}
.welcome { text-align:center; margin: 9vh auto 2rem; max-width: 640px; }
.welcome .hi { font-size: 3rem; font-weight: 800; line-height:1.15;
  background: linear-gradient(90deg,#7C3AED,#EC4899 55%,#F59E0B);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.welcome .sub { font-size:1.15rem; color:#475569; margin-top:.7rem; }
.welcome .neural-mini { width:150px; margin: 0 auto 1rem; display:block; }
.chat-title { font-size: 1.9rem; font-weight: 800; color:#4C1D95; margin-bottom:.2rem; }
.chat-sub { color:#64748B; margin-bottom:1rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#F5F3FF,#ECFEFF); }
.hello-pill {
  display:inline-block; padding:.45rem .9rem; border-radius:999px; font-weight:700;
  background: linear-gradient(90deg,#EDE9FE,#CFFAFE); color:#4C1D95;
}
</style>
"""

SUGGESTIONS = [
    "What is machine learning?",
    "يعني إيه supervised و unsupervised learning؟",
    "What is overfitting and how do I avoid it?",
    "اشرح لي الـ Gradient Descent بأمثلة",
]


# ───────────────────────── Pages ─────────────────────────
def auth_page():
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="brand">🧠 ML Buddy</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title">Learn Machine Learning,<br>one chat at a time.</div>'
            '<div class="hero-sub">Your friendly AI tutor that explains ML simply, in English, '
            'Egyptian Arabic, or Fusha. Whatever suits you.</div>'
            '<div class="chips"><span>Supervised</span><span>Unsupervised</span>'
            '<span>Reinforcement</span><span>Deep Learning</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(NEURAL_SVG, unsafe_allow_html=True)

    with right:
        st.write("")
        st.write("")
        tab_in, tab_up = st.tabs(["🔑 Log in", "✨ Sign up"])

        with tab_in:
            with st.form("login_form"):
                email = st.text_input("Email", key="li_email")
                pw = st.text_input("Password", type="password", key="li_pw")
                ok = st.form_submit_button("Log in", use_container_width=True)
            if ok:
                user, err = log_in(email, pw)
                if err:
                    st.error(err)
                else:
                    start_session(user)

        with tab_up:
            with st.form("signup_form"):
                name = st.text_input("Your name", key="su_name")
                email = st.text_input("Email", key="su_email")
                pw = st.text_input("Password", type="password", key="su_pw")
                ok = st.form_submit_button("Create my account", use_container_width=True)
            if ok:
                user, err = sign_up(name, email, pw)
                if err:
                    st.error(err)
                else:
                    start_session(user)


def start_session(user):
    st.session_state.user_id = user["id"]
    st.session_state.email = user["email"]
    st.session_state.messages = []
    st.rerun()


def logout():
    for k in ["user_id", "email", "messages", "pending"]:
        st.session_state.pop(k, None)
    st.rerun()


def chat_page(user):
    name = display_name(user)

    with st.sidebar:
        st.markdown(f'<div class="brand">🧠 ML Buddy</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="hello-pill">👋 {name}</span>', unsafe_allow_html=True)
        st.caption("Tip: tell me in the chat how you'd like to be called.")
        st.divider()
        st.markdown("**Try asking**")
        for i, s in enumerate(SUGGESTIONS):
            if st.button(s, key=f"sg{i}", use_container_width=True):
                st.session_state.pending = s
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("🧹 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if c2.button("🚪 Logout", use_container_width=True):
            logout()

    if not API_KEY:
        st.error("GEMINI_API_KEY is missing. Add it to .streamlit/secrets.toml or the app's Secrets.")
        return

    # Centered welcome (default English) until the first message
    welcome = st.empty()
    if not st.session_state.messages:
        mini = NEURAL_SVG.replace('class="neural"', 'class="neural-mini"')
        welcome.markdown(
            f'<div class="welcome">{mini}'
            f'<div class="hi">Hello, {name}! 👋</div>'
            '<div class="sub">Everything is up and running. I\'m your Machine Learning assistant. '
            'Ask me anything about ML and I\'ll explain it simply, in English or Arabic.</div></div>',
            unsafe_allow_html=True,
        )

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧠" if m["role"] == "assistant" else "🙂"):
            st.markdown(m["content"])

    prompt = st.chat_input("Type your question...")
    prompt = st.session_state.pop("pending", None) or prompt

    if prompt:
        welcome.empty()
        if not prompt.strip():
            st.warning("Please type a question.")
            return
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        with st.chat_message("user", avatar="🙂"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            raw = []
            try:
                first = len(st.session_state.messages) == 1
                text = st.write_stream(stream_reply(st.session_state.messages, name, raw, first))
            except Exception as e:
                text = f"⚠️ {friendly_error(e)}"
                st.error(text)
                st.session_state.messages.pop()
                return

        st.session_state.messages.append({"role": "assistant", "content": text})

        # LLM asked to change the user's name -> save it for all future sessions
        m = NAME_TAG.search(raw[0]) if raw else None
        if m:
            new_name = re.sub(r"[\[\]<>]", "", m.group(1)).strip()[:40]
            if new_name:
                set_preferred_name(user["id"], new_name)
                st.rerun()


# ───────────────────────── Main ─────────────────────────
init_db()
st.markdown(CSS, unsafe_allow_html=True)

if "user_id" not in st.session_state:
    auth_page()
else:
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (st.session_state.user_id,)).fetchone()
    if row is None:
        logout()
    else:
        chat_page(row)
