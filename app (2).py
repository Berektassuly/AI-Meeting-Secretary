import os
import re
import json
import uuid
import shutil
import tempfile
from datetime import datetime, timezone, date, timedelta
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import imageio_ffmpeg
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = str(os.path.dirname(_FFMPEG_EXE)) + os.pathsep + os.environ.get("PATH","")
except Exception:
    _FFMPEG_EXE = shutil.which("ffmpeg")

try:
    import dateparser
except Exception:
    dateparser = None

st.set_page_config(page_title="Whisper → LLaMA → Jira", page_icon="🌀", layout="wide")

LLAMA_BASE = os.getenv("LLAMA_BASE", "https://vsjz8fv63q4oju-8000.proxy.runpod.net")
LLAMA_URL = os.getenv("LLAMA_URL", "")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "app-V8dsYkzX4pu9544X7qUPrF4J")
LLAMA_AUTH_HEADER = os.getenv("LLAMA_AUTH_HEADER", "Authorization")
LLAMA_AUTH_SCHEME = os.getenv("LLAMA_AUTH_SCHEME", "Bearer")

DEVICE = "cuda" if os.system("nvidia-smi >/dev/null 2>&1") == 0 else "cpu"
COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL_NAME = "medium"

SUPPORTED_UPLOAD_TYPES = ["wav","mp3","m4a","ogg","flac","mp4","mov","mkv","webm"]
MAX_SUMMARY_LEN = 160

def theme_purple_neon():
    css = """
    <style>
    html, body, .stApp { background: #13082A !important; }
    .stApp {
        background: radial-gradient(1250px 400px at 20% 0%, rgba(147,51,234,0.18) 0%, rgba(18,6,38,0) 55%) ,
                    radial-gradient(1250px 400px at 80% 0%, rgba(255,0,153,0.16) 0%, rgba(18,6,38,0) 55%) ,
                    linear-gradient(180deg, #191035 0%, #1D0E3A 50%, #160A2A 100%);
        color: #ECECFF;
    }
    .block-container { max-width: 1228px; padding-top: 2.2rem; padding-bottom: 2rem; }
    h1,h2,h3,h4,h5,h6 { color:#F2ECFF!important; letter-spacing:.3px; }
    [data-testid="stMarkdownContainer"] h1 { margin-top:.2rem; }
    .stAlert { background: rgba(255,255,255,0.06)!important; border:1px solid rgba(255,255,255,0.18)!important; color:#F4F4FD!important; border-radius:16px!important; }
    .stExpander { background: rgba(255,255,255,0.06)!important; border:1px solid rgba(255,255,255,0.14)!important; border-radius:18px!important; }
    .stTextArea textarea, .stTextInput input, .stDateInput input, .stSelectbox div[data-baseweb="select"] input {
        background: rgba(255,255,255,0.08) !important;
        color: #F2F2FF !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 14px !important;
    }
    textarea::placeholder, input::placeholder { color: rgba(245,245,255,0.55) !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.07) !important;
        border-radius: 18px !important;
        border: 1px dashed rgba(255,255,255,0.28) !important;
        color: #ECECFF !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #ECECFF !important; }
    .stFileUploader > div { background: transparent !important; }
    .uploadedFile, .uploadedFile * { background: transparent !important; color: #EDEBFF !important; }
    audio { width: 100%; border-radius: 32px; background: rgba(255,255,255,0.07) !important; }
    audio::-webkit-media-controls-panel { background: rgba(255,255,255,0.08) !important; }
    .stButton button {
        border-radius: 14px;
        padding: .9rem 1.2rem;
        border: 0;
        background: linear-gradient(90deg, #7b2ff7, #f107a3);
        color: #091016;
        font-weight: 700;
        letter-spacing: .2px;
        box-shadow: 0 0 14px rgba(255,0,168,0.45), 0 0 28px rgba(115,0,255,0.35);
        text-shadow: 0 0 6px rgba(255,255,255,0.25);
        transition: transform .1s ease, box-shadow .15s ease;
    }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 0 18px rgba(255,0,168,0.8), 0 0 48px rgba(115,0,255,0.6); }
    .stDownloadButton button { border-radius: 12px; padding: .6rem 1rem; background: linear-gradient(90deg,#00f5d4,#00bbf9); color:#061317; border:0; }
    .neon-card { background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.14); border-radius:20px; padding:14px; }
    .cap { color: #bfbfe9; font-size: .92rem; margin-top: .2rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def to_iso_date(d: Any) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, date):
        return d.isoformat()
    s = str(d).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    return None

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def parse_natural_date(text_value: str) -> str:
    s = (text_value or "").strip().lower()
    if not s:
        return ""
    today = datetime.now().date()
    if s in ("сегодня","today"):
        return today.isoformat()
    if s in ("завтра","tomorrow"):
        return (today + timedelta(days=1)).isoformat()
    if s in ("послезавтра","day after tomorrow"):
        return (today + timedelta(days=2)).isoformat()
    mrel = re.match(r"^через\s+(\d+)\s*(дн(я|ей|ь)?|day|days)$", s)
    if mrel:
        n = int(mrel.group(1))
        return (today + timedelta(days=_clamp(n,0,3650))).isoformat()
    mrelw = re.match(r"^через\s+(\d+)\s*(недел(ю|и|ь|и)|week|weeks)$", s)
    if mrelw:
        n = int(mrelw.group(1))
        return (today + timedelta(days=_clamp(n,0,520)*7)).isoformat()
    mreld = re.match(r"^через\s+(\d+)\s*(месяц(ев)?|month|months)$", s)
    if mreld:
        n = _clamp(int(mreld.group(1)),0,120)
        y, m = today.year, today.month + n
        y += (m - 1) // 12
        m = (m - 1) % 12 + 1
        from datetime import date as dcls
        last = 28
        for ddd in [31,30,29,28]:
            try:
                last = ddd
                dcls(y, m, ddd)
                break
            except Exception:
                continue
        return dcls(y, m, min(today.day, last)).isoformat()
    m_dmy = re.match(r"^(\d{2})[./-](\d{2})[./-](\d{4})$", s)
    if m_dmy:
        dd, mm, yyyy = m_dmy.group(1), m_dmy.group(2), m_dmy.group(3)
        try:
            return date(int(yyyy), int(mm), int(dd)).isoformat()
        except Exception:
            pass
    m_ymd = re.match(r"^(\d{4})[./-](\d{2})[./-](\d{2})$", s)
    if m_ymd:
        yyyy, mm, dd = m_ymd.group(1), m_ymd.group(2), m_ymd.group(3)
        try:
            return date(int(yyyy), int(mm), int(dd)).isoformat()
        except Exception:
            pass
    if dateparser:
        try:
            dt = dateparser.parse(s, languages=["ru","en"], settings={"PREFER_DATES_FROM":"future","RELATIVE_BASE": datetime.now()})
            if dt:
                return dt.date().isoformat()
        except Exception:
            pass
    return ""

def ensure_state():
    st.session_state.setdefault("uploaded_file_name", "")
    st.session_state.setdefault("uploaded_file_bytes", b"")
    st.session_state.setdefault("transcript", "")
    st.session_state.setdefault("transcript_area", "")
    st.session_state.setdefault("tasks", [])
    st.session_state.setdefault("llama_mode", "")
    st.session_state.setdefault("llama_url_used", "")
    st.session_state.setdefault("llama_model_used", "")
    st.session_state.setdefault("lang_hint", "auto")

def stable_id(n: int = 10) -> str:
    return uuid.uuid4().hex[:n]

def extract_audio_to_wav(src_path: str) -> str:
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    ffmpeg = _FFMPEG_EXE or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    cmd = f'"{ffmpeg}" -y -i "{src_path}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{out_path}"'
    code = os.system(cmd)
    if code != 0 or not os.path.exists(out_path):
        raise RuntimeError("ffmpeg failed")
    return out_path

def build_llama_headers() -> Dict[str,str]:
    headers = {"Content-Type": "application/json"}
    token = (LLAMA_AUTH_SCHEME + " " + LLAMA_API_KEY).strip() if LLAMA_API_KEY else ""
    if token:
        headers[LLAMA_AUTH_HEADER] = token
    return headers

def list_models(base: str, headers: Dict[str,str]) -> List[str]:
    try:
        r = requests.get(base.rstrip("/") + "/v1/models", headers=headers, timeout=45)
        if not r.ok:
            return []
        data = r.json().get("data", [])
        out = []
        for m in data:
            mid = m.get("id") if isinstance(m, dict) else None
            if isinstance(mid, str):
                out.append(mid)
        return out
    except Exception:
        return []

def pick_model(models: List[str], prefer: str) -> Optional[str]:
    if prefer and prefer in models:
        return prefer
    if prefer:
        low = prefer.lower()
        for m in models:
            if m.lower() == low:
                return m
    ranked = []
    for m in models:
        ml = m.lower()
        score = 0
        if "instruct" in ml or "chat" in ml: score += 3
        if "llama" in ml: score += 2
        if "scout" in ml: score += 1
        if "8b" in ml: score += 1
        if "fp8" in ml: score += 1
        ranked.append((score, m))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][1] if ranked else None

def try_chat(base: str, headers: Dict[str,str], model: str) -> bool:
    url = base.rstrip("/") + "/v1/chat/completions"
    payload = {"model": model, "messages":[{"role":"user","content":"ping"}], "temperature":0.1}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        return r.status_code == 200
    except Exception:
        return False

def try_responses(base: str, headers: Dict[str,str], model: str) -> bool:
    url = base.rstrip("/") + "/v1/responses"
    payload = {"model": model, "input":[{"role":"user","content":"ping"}], "temperature":0.1}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        return r.status_code == 200
    except Exception:
        return False

def autodiscover_endpoint_and_model() -> Tuple[str,str,str]:
    base = LLAMA_BASE.strip().rstrip("/")
    env_url = LLAMA_URL.strip()
    headers = build_llama_headers()
    models = list_models(base, headers)
    model = pick_model(models, LLAMA_MODEL.strip()) or LLAMA_MODEL.strip() or (models[0] if models else "")
    mode = ""
    url = ""
    if env_url:
        url = env_url
        if "/chat/completions" in url: mode = "chat"
        elif "/responses" in url: mode = "responses"
    else:
        if model and try_chat(base, headers, model):
            mode = "chat"; url = base + "/v1/chat/completions"
        elif model and try_responses(base, headers, model):
            mode = "responses"; url = base + "/v1/responses"
        else:
            if try_chat(base, headers, model or "llama"):
                mode = "chat"; url = base + "/v1/chat/completions"
            elif try_responses(base, headers, model or "llama"):
                mode = "responses"; url = base + "/v1/responses"
    return mode, url, model

def parse_llama_json(text: str) -> List[Dict[str,Any]]:
    m = re.search(r"\[[\s\S]*\]", text)
    js = m.group(0) if m else text
    data = json.loads(js)
    if not isinstance(data, list): raise ValueError("not a list")
    out = []
    for it in data:
        if not isinstance(it, dict): continue
        summary_val = (str(it.get("summary", "")).strip()[:MAX_SUMMARY_LEN])
        desc_val = str(it.get("description", "")).strip()
        labels_raw = str(it.get("labels", ""))
        parts = [p.strip() for p in labels_raw.split(",") if p.strip()]
        parts = [p for p in parts if not re.match(r"(?i)^category\s*:", p)]
        due_raw = str(it.get("due", "")).strip()
        due_iso = parse_natural_date(due_raw) or to_iso_date(due_raw) or "" if due_raw else ""
        comment_val = str(it.get("comment", ""))
        out.append({"id": stable_id(), "summary": summary_val, "description": desc_val, "labels": ", ".join(parts), "due": due_iso, "comment": comment_val})
    return out

def llama_call(mode: str, url: str, model: str, messages: List[Dict[str,str]]) -> str:
    headers = build_llama_headers()
    if mode == "chat":
        payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 4000}
    else:
        payload = {"model": model, "input": messages, "temperature": 0.1, "max_tokens": 4000}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    if mode == "chat":
        return data["choices"][0]["message"]["content"]
    out = data.get("output_text")
    if isinstance(out, str) and out.strip(): return out
    outputs = data.get("output", []) or data.get("choices", [])
    if outputs and isinstance(outputs, list):
        first = outputs[0]
        return first.get("content") or first.get("message", {}).get("content") or ""
    return ""

def llama_clean_text(raw_text: str) -> Tuple[str, Dict[str,str]]:
    mode, url, model = autodiscover_endpoint_and_model()
    if not url or not mode: raise RuntimeError("LLM endpoint not found")
    sys = "Ты редактор текста. Исправь опечатки и пунктуацию, не меняй смысл. Верни только исправленный текст."
    content = llama_call(mode, url, model, [{"role":"system","content":sys},{"role":"user","content":raw_text}])
    meta = {"mode": mode, "url": url, "model": model}
    return content.strip(), meta

def llama_extract_tasks(transcript: str) -> Tuple[List[Dict[str,Any]], Dict[str,str]]:
    mode, url, model = autodiscover_endpoint_and_model()
    if not url or not mode: raise RuntimeError("LLM endpoint not found")
    sys = "Верни только JSON-массив задач. Ключи: summary, description, labels, due, comment. Никаких пояснений."
    txt = llama_call(mode, url, model, [{"role":"system","content":sys},{"role":"user","content":transcript}])
    tasks = parse_llama_json(txt)
    meta = {"mode": mode, "url": url, "model": model}
    return tasks, meta

def jira_create_issue(base_url: str, email: str, api_token: str, project_key: str, task: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/rest/api/3/issue"
    headers = {"Accept":"application/json","Content-Type":"application/json"}
    raw = task.get("labels","") or ""
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    due = to_iso_date(task.get("due"))
    fields = {"project":{"key":project_key},"summary":(task.get("summary") or "Задача")[:MAX_SUMMARY_LEN],"issuetype":{"name":"Task"}}
    if labels: fields["labels"] = labels
    if due: fields["duedate"] = due
    desc = task.get("description","").strip()
    if desc: fields["description"] = {"type":"doc","version":1,"content":[{"type":"paragraph","content":[{"type":"text","text":desc}]}]}
    payload = {"fields": fields}
    r = requests.post(url, auth=(email, api_token), json=payload, headers=headers, timeout=60)
    if r.status_code >= 300: return {"ok": False, "error": r.text}
    return {"ok": True, **r.json()}

def jira_add_comment(base_url: str, email: str, api_token: str, issue_key: str, text: str) -> Dict[str, Any]:
    if not (text or "").strip(): return {"ok": True, "skipped": True}
    url = base_url.rstrip("/") + f"/rest/api/3/issue/{issue_key}/comment"
    headers = {"Accept":"application/json","Content-Type":"application/json"}
    body = {"body": {"type":"doc","version":1,"content":[{"type":"paragraph","content":[{"type":"text","text":text}]}]}}
    r = requests.post(url, auth=(email, api_token), json=body, headers=headers, timeout=60)
    if r.status_code >= 300: return {"ok": False, "error": r.text}
    return {"ok": True, **r.json()}

def issue_link(base_url: str, key: str) -> str:
    return base_url.rstrip("/") + "/browse/" + key

def project_list_link(base_url: str, key: str) -> str:
    return base_url.rstrip("/") + "/jira/core/projects/" + key + "/list"

ensure_state()
theme_purple_neon()

@st.cache_resource(show_spinner=True)
def load_whisper() -> WhisperModel:
    return WhisperModel(WHISPER_MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

if WhisperModel is None:
    st.error("faster-whisper не установлен")
    st.stop()

whisper_model = load_whisper()

st.title("Whisper → LLaMA → Jira")

st.header("Загрузка и распознавание")
file = st.file_uploader("Форматы: wav, mp3, m4a, ogg, flac, mp4, mov, mkv, webm", type=SUPPORTED_UPLOAD_TYPES)
if file is not None:
    st.session_state["uploaded_file_name"] = file.name
    st.session_state["uploaded_file_bytes"] = file.getvalue()
if st.session_state.get("uploaded_file_bytes"):
    try:
        st.audio(st.session_state["uploaded_file_bytes"])
    except Exception:
        pass

st.header("Распознать")
c1, c2, _ = st.columns([1,1,6], gap="medium")
with c1:
    btn_rec = st.button("Распознать из аудио", type="primary")
with c2:
    lang_opt = st.selectbox("Язык", ["auto","ru","en","kk","tr"], index=["auto","ru","en","kk","tr"].index(st.session_state.get("lang_hint","auto")))
    st.session_state["lang_hint"] = lang_opt

if btn_rec:
    if not st.session_state.get("uploaded_file_bytes"):
        st.warning("Сначала загрузите файл")
    else:
        tmp_src = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{st.session_state['uploaded_file_name']}")
        tmp_src.write(st.session_state["uploaded_file_bytes"])
        tmp_src.flush()
        tmp_src.close()
        src_path = tmp_src.name
        ext = (st.session_state["uploaded_file_name"].split(".")[-1] or "").lower()
        audio_path = src_path
        if ext in ["mp4","mov","mkv","webm"]:
            audio_path = extract_audio_to_wav(src_path)
        try:
            st.info("Распознаю, подождите...")
            kwargs = {}
            hint = st.session_state.get("lang_hint","auto")
            if hint and hint != "auto": kwargs["language"] = hint
            parts = []
            segments, info = whisper_model.transcribe(audio_path, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500}, **kwargs)
            for seg in segments: parts.append(seg.text)
            raw_text = "".join(parts).strip()
            cleaned, meta = llama_clean_text(raw_text) if raw_text else ("", {})
            final_text = cleaned or raw_text
            st.session_state["transcript"] = final_text
            st.session_state["transcript_area"] = final_text
            st.session_state["llama_mode"] = meta.get("mode","")
            st.session_state["llama_url_used"] = meta.get("url","")
            st.session_state["llama_model_used"] = meta.get("model","")
            st.success("Готово")
        finally:
            try: os.unlink(src_path)
            except Exception: pass
            try:
                if audio_path != src_path: os.unlink(audio_path)
            except Exception: pass

st.header("Распознанный текст")
txt_placeholder = "Тут появится распознанный и автоматически исправленный текст"
current_text = st.session_state.get("transcript_area") or st.session_state.get("transcript") or ""
edited_area = st.text_area("Текст", value=current_text, placeholder=txt_placeholder, height=280, key="transcript_area_widget")

st.header("Извлечение")
btn_extract = st.button("Извлечь задачи", type="secondary")
if btn_extract:
    text_for_llama = edited_area or st.session_state.get("transcript") or ""
    if not text_for_llama.strip():
        st.warning("Нет текста для извлечения")
    else:
        try:
            tasks, meta2 = llama_extract_tasks(text_for_llama)
            st.session_state["tasks"] = tasks
            st.session_state["llama_mode"] = meta2.get("mode","")
            st.session_state["llama_url_used"] = meta2.get("url","")
            st.session_state["llama_model_used"] = meta2.get("model","")
            st.success(f"Извлечено задач: {len(tasks)}")
        except Exception as e:
            st.error(str(e))

st.header("Список задач")
new_list = []
if st.session_state.get("tasks"):
    if st.button("Нормализовать дедлайны", type="secondary"):
        for t in st.session_state["tasks"]:
            raw_due = t.get("due","").strip()
            if raw_due: t["due"] = parse_natural_date(raw_due) or to_iso_date(raw_due) or raw_due
    idx = 1
    for t in st.session_state.get("tasks", []):
        with st.expander(f"Задача {idx}: {t.get('summary') or ''}".strip(), expanded=False):
            t["summary"] = st.text_input("Тема", t.get("summary",""), key=f"s_{t['id']}")
            t["description"] = st.text_area("Описание", t.get("description",""), key=f"d_{t['id']}")
            t["labels"] = st.text_input("Метки (через запятую)", t.get("labels",""), key=f"l_{t['id']}")
            t["due"] = st.text_input("Дата дедлайна", value=t.get("due",""), placeholder="YYYY-MM-DD пример: 2025-09-21", key=f"due_{t['id']}")
            t["comment"] = st.text_area("Комментарий", t.get("comment",""), key=f"c_{t['id']}")
            if st.button("Удалить", key=f"del_{t['id']}"):
                pass
            else:
                new_list.append(t)
        idx += 1
st.session_state["tasks"] = new_list

if st.button("Добавить задачу вручную"):
    st.session_state["tasks"].append({"id": stable_id(), "summary":"", "description":"", "labels":"", "due":"", "comment":""})

st.header("Отправка в Jira")
with st.form("jira_form", clear_on_submit=False):
    jira_url = st.text_input("Jira URL", placeholder="https://your-domain.atlassian.net", key="jira_url")
    jira_email = st.text_input("Jira Email", key="jira_email")
    jira_token = st.text_input("Jira API Token", type="password", key="jira_token")
    jira_project = st.text_input("Project Key", placeholder="PRJ", key="jira_project")
    btn_submit = st.form_submit_button("Создать задачи", type="primary")

js_enter = """
<script>
(function(){
  function enhanceForm(){
    const form = Array.from(document.querySelectorAll('form')).slice(-1)[0];
    if(!form) { setTimeout(enhanceForm, 500); return; }
    const inputs = form.querySelectorAll('input, textarea');
    inputs.forEach((el, idx) => {
      el.addEventListener('keydown', function(e){
        if(e.key === 'Enter' && !e.shiftKey){
          e.preventDefault();
          const next = inputs[idx+1];
          if(next){ next.focus(); }
          else {
            const btns = Array.from(form.querySelectorAll('button'));
            const target = btns.find(b => b.innerText.trim().includes('Создать задачи'));
            if(target){ target.click(); }
          }
        }
      });
    });
  }
  setTimeout(enhanceForm, 700);
})();
</script>
"""
st.markdown(js_enter, unsafe_allow_html=True)

def jira_create_all(jira_url, jira_email, jira_token, jira_project, tasks: List[Dict[str,Any]]):
    oks = []; errs = []; links = []
    for t in tasks:
        due_raw = (t.get("due") or "").strip()
        if due_raw and not re.match(r"^\d{4}-\d{2}-\d{2}$", due_raw): t["due"] = parse_natural_date(due_raw) or ""
        res = jira_create_issue(jira_url, jira_email, jira_token, jira_project, t)
        if not res.get("ok"):
            errs.append(res.get("error","")); continue
        key = res.get("key") or res.get("id") or "?"
        if (t.get("comment") or "").strip():
            _r = jira_add_comment(jira_url, jira_email, jira_token, key, t["comment"].strip())
            if not _r.get("ok"): errs.append(_r.get("error",""))
        oks.append(key); links.append(issue_link(jira_url, key))
    return oks, errs, links

if btn_submit:
    miss = [x for x, v in [("URL", st.session_state.get("jira_url")), ("Email", st.session_state.get("jira_email")), ("API token", st.session_state.get("jira_token")), ("Project Key", st.session_state.get("jira_project"))] if not (v or "").strip()]
    if miss:
        st.warning("Заполните поля: " + ", ".join(miss))
    else:
        tlist = list(st.session_state.get("tasks", []))
        if not tlist:
            st.error("Нет задач для отправки")
        else:
            created, errors, lks = jira_create_all(st.session_state["jira_url"], st.session_state["jira_email"], st.session_state["jira_token"], st.session_state["jira_project"], tlist)
            if created:
                st.success("Создано: " + ", ".join(created))
                proj_url = project_list_link(st.session_state["jira_url"], st.session_state["jira_project"])
                st.write("Проект " + st.session_state["jira_project"] + ": " + proj_url)
                for u in lks: st.write(u)
            if errors: st.error("Ошибки: " + " | ".join([e[:200] for e in errors]))
