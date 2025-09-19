import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - fallback when package missing
    WhisperModel = None

try:
    import imageio_ffmpeg

    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = (
        str(os.path.dirname(_FFMPEG_EXE)) + os.pathsep + os.environ.get("PATH", "")
    )
except Exception:
    _FFMPEG_EXE = shutil.which("ffmpeg")

try:
    import dateparser
except Exception:
    dateparser = None

st.set_page_config(page_title="Whisper → LLaMA → Jira", page_icon="🌀", layout="wide")

LLAMA_BASE = os.getenv("LLAMA_BASE", "").strip()
LLAMA_URL = os.getenv("LLAMA_URL", "").strip()
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "").strip()
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "").strip()
LLAMA_AUTH_HEADER = os.getenv("LLAMA_AUTH_HEADER", "Authorization")
LLAMA_AUTH_SCHEME = os.getenv("LLAMA_AUTH_SCHEME", "Bearer")

# Probe GPU availability without invoking shell redirects that fail on Windows.
def _probe_nvidia_smi() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
    except Exception:
        return False
    return result.returncode == 0


_HAS_NVIDIA_SMI = _probe_nvidia_smi()
_DEFAULT_DEVICE = "cuda" if _HAS_NVIDIA_SMI else "cpu"
WHISPER_MODEL_CPU = os.getenv("WHISPER_MODEL_CPU", "small")
WHISPER_MODEL_CUDA = os.getenv("WHISPER_MODEL_CUDA", "medium")


def _normalize_device_name(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    return normalized if normalized in {"cpu", "cuda"} else ""

SUPPORTED_UPLOAD_TYPES = [
    "wav",
    "mp3",
    "m4a",
    "ogg",
    "flac",
    "mp4",
    "mov",
    "mkv",
    "webm",
]
VIDEO_EXTENSIONS = {"mp4", "mov", "mkv", "webm"}
PRIORITIES = ["Highest", "High", "Medium", "Low", "Lowest"]
MAX_SUMMARY = 160
def apply_theme() -> None:
    css = """
    <style>
    html, body, .stApp { background: #13082A !important; }
    .stApp {
        background:
            radial-gradient(1250px 400px at 20% 0%, rgba(147,51,234,0.18) 0%, rgba(18,6,38,0) 55%),
            radial-gradient(1250px 400px at 80% 0%, rgba(255,0,153,0.16) 0%, rgba(18,6,38,0) 55%),
            linear-gradient(180deg, #191035 0%, #1D0E3A 50%, #160A2A 100%);
        color: #ECECFF;
    }
    .block-container { max-width: 1228px; padding-top: 2.2rem; padding-bottom: 2rem; }
    h1,h2,h3,h4,h5,h6 { color:#F2ECFF!important; letter-spacing:.3px; }
    [data-testid=stMarkdownContainer] h1 { margin-top:.2rem; }
    .stAlert {
        background: rgba(255,255,255,0.06)!important;
        border:1px solid rgba(255,255,255,0.18)!important;
        color:#F4F4FD!important;
        border-radius:16px!important;
    }
    .stExpander {
        background: rgba(255,255,255,0.06)!important;
        border:1px solid rgba(255,255,255,0.14)!important;
        border-radius:18px!important;
    }
    .stTextArea textarea,
    .stTextInput input,
    .stDateInput input,
    div[data-baseweb=select] input {
        background: rgba(255,255,255,0.08) !important;
        color: #F2F2FF !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 14px !important;
    }
    textarea::placeholder, input::placeholder {
        color: rgba(245,245,255,0.55) !important;
    }
    [data-testid=stFileUploaderDropzone] {
        background: rgba(255,255,255,0.07) !important;
        border-radius: 18px !important;
        border: 1px dashed rgba(255,255,255,0.28) !important;
        color: #ECECFF !important;
    }
    [data-testid=stFileUploaderDropzone] * { color: #ECECFF !important; }
    .stFileUploader > div { background: transparent !important; }
    .uploadedFile, .uploadedFile * {
        background: transparent !important;
        color: #EDEBFF !important;
    }
    audio {
        width: 100%;
        border-radius: 32px;
        background: rgba(255,255,255,0.07) !important;
    }
    audio::-webkit-media-controls-panel {
        background: rgba(255,255,255,0.08) !important;
    }
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
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 18px rgba(255,0,168,0.8), 0 0 48px rgba(115,0,255,0.6);
    }
    .stDownloadButton button {
        border-radius: 12px;
        padding: .6rem 1rem;
        background: linear-gradient(90deg,#00f5d4,#00bbf9);
        color:#061317;
        border:0;
    }
    .neon-card {
        background: rgba(255,255,255,0.06);
        border:1px solid rgba(255,255,255,0.14);
        border-radius:20px;
        padding:14px;
    }
    .cap { color: #bfbfe9; font-size: .92rem; margin-top: .2rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def ensure_state() -> None:
    defaults = {
        "uploaded_file_name": "",
        "uploaded_file_bytes": b"",
        "transcript": "",
        "transcript_area": "",
        "tasks": [],
        "lang_hint": "auto",
        "llm_meta": {"mode": "", "url": "", "model": ""},
        "whisper_meta": {
            "device": "",
            "compute_type": "",
            "model": "",
            "fallback": False,
        },
        "whisper_force_device": "",
        "metrics": {
            "extract_audio_ms": None,
            "asr_ms": None,
            "llm_clean_ms": None,
            "extract_tasks_ms": None,
            "jira_create_ms": None,
            "tasks_total": 0,
            "tasks_created": 0,
            "errors_count": 0,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def stable_id(length: int = 10) -> str:
    return uuid.uuid4().hex[:length]


def autolabels_from_summary(text: str, limit: int = 6) -> str:
    words = re.findall(r"[\w\-А-Яа-яЁё]{3,}", text or "", flags=re.UNICODE)
    seen: set[str] = set()
    labels: List[str] = []
    for word in words:
        low = word.lower()
        if low in seen:
            continue
        seen.add(low)
        labels.append(low)
        if len(labels) >= max(3, limit):
            break
    return ", ".join(labels)

def to_iso_date(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}$', text):
        return text
    return ''


_MONTH_MAP = {
    'январь': 1,
    'января': 1,
    'янв': 1,
    'feb': 2,
    'фев': 2,
    'февраль': 2,
    'февраля': 2,
    'march': 3,
    'mar': 3,
    'мар': 3,
    'март': 3,
    'марта': 3,
    'apr': 4,
    'апр': 4,
    'апрель': 4,
    'апреля': 4,
    'may': 5,
    'май': 5,
    'мая': 5,
    'июнь': 6,
    'июня': 6,
    'jun': 6,
    'june': 6,
    'июль': 7,
    'июля': 7,
    'jul': 7,
    'july': 7,
    'aug': 8,
    'august': 8,
    'август': 8,
    'августа': 8,
    'sep': 9,
    'sept': 9,
    'september': 9,
    'сен': 9,
    'сентябрь': 9,
    'сентября': 9,
    'oct': 10,
    'october': 10,
    'октябрь': 10,
    'окт': 10,
    'октября': 10,
    'nov': 11,
    'november': 11,
    'ноябрь': 11,
    'ноября': 11,
    'dec': 12,
    'december': 12,
    'декабрь': 12,
    'декабря': 12,
}


def _normalize_year(value: int) -> int:
    if value < 100:
        return 2000 + value if value < 80 else 1900 + value
    return value


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _safe_date(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ''


def parse_natural_date(text_value: str) -> str:
    text = (text_value or '').strip()
    if not text:
        return ''
    lower = text.lower()
    today = date.today()

    special = {
        'сегодня': today,
        'today': today,
        'завтра': today + timedelta(days=1),
        'tomorrow': today + timedelta(days=1),
        'послезавтра': today + timedelta(days=2),
        'day after tomorrow': today + timedelta(days=2),
    }
    if lower in special:
        return special[lower].isoformat()

    match_days = re.match(r'^через\s+(\d+)\s*(дн(я|ей|ь)?|day|days)$', lower)
    if match_days:
        days = int(match_days.group(1))
        return (today + timedelta(days=max(0, min(days, 3650)))).isoformat()

    match_weeks = re.match(r'^через\s+(\d+)\s*(недел(ю|и|ь|и)|week|weeks)$', lower)
    if match_weeks:
        weeks = int(match_weeks.group(1))
        return (today + timedelta(days=max(0, min(weeks, 520)) * 7)).isoformat()

    match_months = re.match(r'^через\s+(\d+)\s*(месяц(ев)?|month|months)$', lower)
    if match_months:
        months = max(0, min(int(match_months.group(1)), 120))
        year = today.year
        month = today.month + months
        year += (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(today.day, _last_day_of_month(year, month))
        return date(year, month, day).isoformat()

    textual = re.match(
        r'^(\d{1,2})\s+([а-яёa-z]+)(?:\s+(\d{2,4}))?$',
        lower,
        flags=re.IGNORECASE,
    )
    if textual:
        dd = int(textual.group(1))
        mm = _MONTH_MAP.get(textual.group(2).lower())
        if mm:
            year_part = textual.group(3)
            year_val = _normalize_year(int(year_part)) if year_part else today.year
            return _safe_date(year_val, mm, dd)

    textual_alt = re.match(
        r'^([а-яёa-z]+)\s+(\d{1,2})(?:\s*,?\s*(\d{2,4}))?$',
        lower,
        flags=re.IGNORECASE,
    )
    if textual_alt:
        mm = _MONTH_MAP.get(textual_alt.group(1).lower())
        dd = int(textual_alt.group(2))
        if mm:
            year_part = textual_alt.group(3)
            year_val = _normalize_year(int(year_part)) if year_part else today.year
            return _safe_date(year_val, mm, dd)

    textual_ord = re.match(
        r'^(\d{1,2})(st|nd|rd|th)?\s+of\s+([a-z]+)(?:\s+(\d{2,4}))?$',
        lower,
        flags=re.IGNORECASE,
    )
    if textual_ord:
        dd = int(textual_ord.group(1))
        mm = _MONTH_MAP.get(textual_ord.group(3).lower())
        if mm:
            year_part = textual_ord.group(4)
            year_val = _normalize_year(int(year_part)) if year_part else today.year
            return _safe_date(year_val, mm, dd)

    patterns = [
        r'^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$',
        r'^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$',
        r'^(\d{1,2})[./-](\d{1,2})[./-](\d{2})$',
    ]
    for pat in patterns:
        found = re.match(pat, lower)
        if not found:
            continue
        g1, g2, g3 = found.groups()
        try:
            if pat.startswith('^(\\d{4})'):
                year_val = int(g1)
                month_val = int(g2)
                day_val = int(g3)
            else:
                day_val = int(g1)
                month_val = int(g2)
                year_val = int(g3)
                if len(g3) == 2:
                    year_val = _normalize_year(year_val)
            return _safe_date(year_val, month_val, day_val)
        except ValueError:
            continue

    if dateparser:
        try:
            parsed = dateparser.parse(
                text,
                languages=['ru', 'en'],
                settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': datetime.now()},
            )
            if parsed:
                return parsed.date().isoformat()
        except Exception:
            return ''

    return ''


def normalize_due_value(raw: str) -> str:
    raw = (raw or '').strip()
    if not raw:
        return ''
    parsed = parse_natural_date(raw)
    if parsed:
        return parsed
    iso = to_iso_date(raw)
    return iso or ''


def extract_audio_to_wav(src_path: str) -> str:
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    ffmpeg = _FFMPEG_EXE or shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg not found')
    cmd = f'"{ffmpeg}" -y -i "{src_path}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{out_path}"'
    code = os.system(cmd)
    if code != 0 or not os.path.exists(out_path):
        raise RuntimeError('ffmpeg failed')
    return out_path

def build_llama_headers() -> Dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    if LLAMA_API_KEY:
        headers[LLAMA_AUTH_HEADER] = f'{LLAMA_AUTH_SCHEME} {LLAMA_API_KEY}'.strip()
    return headers


def list_models(base: str, headers: Dict[str, str]) -> List[str]:
    try:
        response = requests.get(base.rstrip('/') + '/v1/models', headers=headers, timeout=30)
        if not response.ok:
            return []
        data = response.json().get('data', [])
        items: List[str] = []
        for entry in data:
            model_id = entry.get('id') if isinstance(entry, dict) else None
            if isinstance(model_id, str):
                items.append(model_id)
        return items
    except Exception:
        return []


def pick_model(models: List[str], prefer: str) -> Optional[str]:
    if prefer and prefer in models:
        return prefer
    if prefer:
        lower_prefer = prefer.lower()
        for model in models:
            if model.lower() == lower_prefer:
                return model
    ranked: List[Tuple[int, str]] = []
    for model in models:
        mark = 0
        low = model.lower()
        if 'instruct' in low or 'chat' in low:
            mark += 3
        if 'llama' in low:
            mark += 2
        if 'scout' in low:
            mark += 1
        if '8b' in low:
            mark += 1
        if 'fp8' in low:
            mark += 1
        ranked.append((mark, model))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1] if ranked else None


def try_chat(base: str, headers: Dict[str, str], model: str) -> bool:
    url = base.rstrip('/') + '/v1/chat/completions'
    payload = {'model': model, 'messages': [{'role': 'user', 'content': 'ping'}], 'temperature': 0.1}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        return response.status_code == 200
    except Exception:
        return False


def try_responses(base: str, headers: Dict[str, str], model: str) -> bool:
    url = base.rstrip('/') + '/v1/responses'
    payload = {'model': model, 'input': [{'role': 'user', 'content': 'ping'}], 'temperature': 0.1}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        return response.status_code == 200
    except Exception:
        return False

def autodiscover_endpoint_and_model() -> Tuple[str, str, str]:
    headers = build_llama_headers()
    if LLAMA_URL:
        raw_url = LLAMA_URL.rstrip('/')
        if '/chat/completions' in raw_url:
            return 'chat', raw_url, LLAMA_MODEL or ''
        if '/responses' in raw_url:
            return 'responses', raw_url, LLAMA_MODEL or ''
        model_hint = LLAMA_MODEL or 'llama'
        if try_chat(raw_url, headers, model_hint):
            return 'chat', raw_url.rstrip('/') + '/v1/chat/completions', model_hint
        if try_responses(raw_url, headers, model_hint):
            return 'responses', raw_url.rstrip('/') + '/v1/responses', model_hint
        raise RuntimeError('LLM endpoint not found')
    if not LLAMA_BASE:
        raise RuntimeError('LLM endpoint not found')
    base = LLAMA_BASE.rstrip('/')
    models = list_models(base, headers)
    preferred = pick_model(models, LLAMA_MODEL) or (LLAMA_MODEL or (models[0] if models else ''))
    candidates = [c for c in [preferred, models[0] if models else None, 'llama'] if c]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if try_chat(base, headers, candidate):
            return 'chat', base + '/v1/chat/completions', candidate
        if try_responses(base, headers, candidate):
            return 'responses', base + '/v1/responses', candidate
    raise RuntimeError('LLM endpoint not found')


def llama_call(mode: str, url: str, model: str, messages: List[Dict[str, str]]) -> str:
    headers = build_llama_headers()
    if mode == 'chat':
        payload = {'model': model, 'messages': messages, 'temperature': 0.1, 'max_tokens': 4000}
    else:
        payload = {'model': model, 'input': messages, 'temperature': 0.1, 'max_tokens': 4000}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f'LLM request failed: {exc}') from exc
    if response.status_code >= 400:
        raise RuntimeError(f'LLM error {response.status_code}: {response.text[:200]}')
    data = response.json()
    if mode == 'chat':
        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
    if 'output_text' in data and isinstance(data['output_text'], str):
        if data['output_text'].strip():
            return data['output_text']
    outputs = data.get('output') or data.get('choices') or []
    if outputs and isinstance(outputs, list):
        first = outputs[0]
        if isinstance(first, dict):
            return first.get('content') or first.get('message', {}).get('content') or ''
    return ''

def extract_json_array(text: str) -> str:
    matches = list(re.finditer(r'\[[\s\S]*\]', text))
    if not matches:
        return text
    longest = max(matches, key=lambda item: len(item.group(0)))
    return longest.group(0)


def parse_tasks_json(text: str) -> List[Dict[str, Any]]:
    blob = extract_json_array(text)
    data = json.loads(blob)
    if not isinstance(data, list):
        raise ValueError('not list')
    tasks: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        summary = str(item.get('summary', '')).strip()[:MAX_SUMMARY]
        description = str(item.get('description', '')).strip()
        labels_field = item.get('labels', '')
        if isinstance(labels_field, list):
            labels_list = [str(part).strip() for part in labels_field if str(part).strip()]
        else:
            labels_list = [part.strip() for part in str(labels_field).split(',') if part.strip()]
        if not labels_list and summary:
            auto = autolabels_from_summary(summary)
            if auto:
                labels_list = [part.strip() for part in auto.split(',') if part.strip()]
        due_raw = str(item.get('due', '')).strip()
        due_iso = normalize_due_value(due_raw) if due_raw else ''
        comment = str(item.get('comment', '')).strip()
        priority_raw = str(item.get('priority', '') or 'Medium').title()
        if priority_raw not in PRIORITIES:
            priority_raw = 'Medium'
        task = {
            'id': stable_id(8),
            'summary': summary,
            'description': description,
            'labels': ', '.join(labels_list),
            'due': due_iso,
            'comment': comment,
            'priority': priority_raw,
        }
        tasks.append(task)
    return tasks


def llama_clean_text(raw_text: str) -> Tuple[str, Dict[str, str]]:
    mode, url, model = autodiscover_endpoint_and_model()
    if not mode or not url:
        raise RuntimeError('LLM endpoint not found')
    system_prompt = 'Ты редактор текста. Исправь опечатки и пунктуацию, не меняй смысл. Верни только исправленный текст.'
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': raw_text},
    ]
    cleaned = llama_call(mode, url, model, messages).strip()
    meta = {'mode': mode, 'url': url, 'model': model}
    return cleaned or raw_text, meta


def llama_extract_tasks(transcript: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    mode, url, model = autodiscover_endpoint_and_model()
    if not mode or not url:
        raise RuntimeError('LLM endpoint not found')
    system_prompt = (
        'Ты помощник по управлению проектами. '
        'Верни ТОЛЬКО JSON-массив задач с полями summary, description, labels, due, comment, priority. '
        'labels — 3–6 ключевых слов через запятую. '
        'priority должно быть одним из Highest, High, Medium, Low, Lowest. '
        'Если дедлайна нет, оставь due пустым. '
        'Дедлайны можно указывать словами вроде "завтра" или "через 3 недели".'
    )
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': transcript},
    ]
    text = llama_call(mode, url, model, messages)
    tasks = parse_tasks_json(text)
    meta = {'mode': mode, 'url': url, 'model': model}
    return tasks, meta

def jira_priority_map(base_url: str, email: str, api_token: str) -> Dict[str, str]:
    try:
        response = requests.get(
            base_url.rstrip('/') + '/rest/api/3/priority',
            auth=(email, api_token),
            timeout=40,
        )
        if response.status_code >= 300:
            return {}
        priorities = {}
        for item in response.json():
            name = str(item.get('name', '')).lower()
            pid = item.get('id')
            if name and pid:
                priorities[name] = str(pid)
        return priorities
    except Exception:
        return {}


def jira_create_issue(
    base_url: str,
    email: str,
    api_token: str,
    project_key: str,
    task: Dict[str, Any],
    priority_map: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    url = base_url.rstrip('/') + '/rest/api/3/issue'
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    fields: Dict[str, Any] = {
        'project': {'key': project_key},
        'summary': (task.get('summary') or 'Задача')[:MAX_SUMMARY],
        'issuetype': {'name': 'Task'},
    }
    labels_raw = task.get('labels', '') or ''
    labels = [label.strip() for label in labels_raw.split(',') if label.strip()]
    if labels:
        fields['labels'] = labels
    due_iso = normalize_due_value(task.get('due', ''))
    if due_iso:
        fields['duedate'] = due_iso
    priority_name = str(task.get('priority') or 'Medium').title()
    if priority_name not in PRIORITIES:
        priority_name = 'Medium'
    if priority_map:
        priority_id = priority_map.get(priority_name.lower())
        if priority_id:
            fields['priority'] = {'id': priority_id}
        else:
            fields['priority'] = {'name': priority_name}
    else:
        fields['priority'] = {'name': priority_name}
    description = str(task.get('description', '')).strip()
    if description:
        fields['description'] = {
            'type': 'doc',
            'version': 1,
            'content': [
                {'type': 'paragraph', 'content': [{'type': 'text', 'text': description}]}
            ],
        }
    payload = {'fields': fields}
    try:
        response = requests.post(
            url,
            auth=(email, api_token),
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException as exc:
        return {'ok': False, 'error': str(exc)}
    if response.status_code >= 300:
        return {'ok': False, 'error': response.text}
    return {'ok': True, **response.json()}


def jira_add_comment(
    base_url: str,
    email: str,
    api_token: str,
    issue_key: str,
    text: str,
) -> Dict[str, Any]:
    if not (text or '').strip():
        return {'ok': True, 'skipped': True}
    url = base_url.rstrip('/') + f'/rest/api/3/issue/{issue_key}/comment'
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    payload = {
        'body': {
            'type': 'doc',
            'version': 1,
            'content': [
                {'type': 'paragraph', 'content': [{'type': 'text', 'text': text}]}
            ],
        }
    }
    try:
        response = requests.post(
            url,
            auth=(email, api_token),
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException as exc:
        return {'ok': False, 'error': str(exc)}
    if response.status_code >= 300:
        return {'ok': False, 'error': response.text}
    return {'ok': True, **response.json()}


def issue_link(base_url: str, key: str) -> str:
    return base_url.rstrip('/') + '/browse/' + key


def project_list_link(base_url: str, key: str) -> str:
    return base_url.rstrip('/') + f'/jira/core/projects/{key}/list'


def jira_create_all(
    base_url: str,
    email: str,
    api_token: str,
    project_key: str,
    tasks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    base = base_url.rstrip('/')
    priority_map = jira_priority_map(base, email, api_token)
    successes: List[str] = []
    errors: List[str] = []
    links: List[str] = []
    for task in tasks:
        payload = dict(task)
        payload['due'] = normalize_due_value(payload.get('due', ''))
        result = jira_create_issue(base, email, api_token, project_key, payload, priority_map)
        if not result.get('ok'):
            errors.append(result.get('error', ''))
            continue
        key = result.get('key') or result.get('id') or '?'
        comment_text = str(task.get('comment', '')).strip()
        if comment_text:
            comment_result = jira_add_comment(base, email, api_token, key, comment_text)
            if not comment_result.get('ok'):
                errors.append(comment_result.get('error', ''))
        successes.append(key)
        links.append(issue_link(base, key))
    return successes, errors, links

def _format_ms(value: Optional[float]) -> str:
    if isinstance(value, (int, float)):
        return f'{value:.0f}'
    return '—'


def _format_count(value: Optional[Any]) -> str:
    if isinstance(value, (int, float)):
        return str(int(value))
    return '0'


def render_metrics() -> None:
    metrics = st.session_state.get('metrics', {})
    st.subheader('Мини-панель метрик')
    durations = [
        ('extract_audio_ms', 'Извлечение аудио (мс)'),
        ('asr_ms', 'Транскрипция (мс)'),
        ('llm_clean_ms', 'Чистка LLM (мс)'),
        ('extract_tasks_ms', 'Извлечение задач (мс)'),
        ('jira_create_ms', 'Создание в Jira (мс)'),
    ]
    cols = st.columns(len(durations))
    for col, (key, label) in zip(cols, durations):
        col.metric(label, _format_ms(metrics.get(key)))
    counts = [
        ('tasks_total', 'Задач извлечено'),
        ('tasks_created', 'Задач создано'),
        ('errors_count', 'Ошибок'),
    ]
    count_cols = st.columns(len(counts))
    for col, (key, label) in zip(count_cols, counts):
        col.metric(label, _format_count(metrics.get(key)))
    meta = st.session_state.get('llm_meta', {})
    if meta.get('url'):
        st.caption(
            f"LLM режим: {meta.get('mode', '-')}, модель: {meta.get('model', '-')}, URL: {meta.get('url')}"
        )
    whisper_meta = st.session_state.get('whisper_meta', {})
    if whisper_meta.get('device'):
        note = ' (авто-переключение на CPU)' if whisper_meta.get('fallback') else ''
        st.caption(
            "Whisper: устройство {device}, тип вычислений {ctype}, модель {model}{note}".format(
                device=whisper_meta.get('device', '-'),
                ctype=whisper_meta.get('compute_type', '-'),
                model=whisper_meta.get('model', '-'),
                note=note,
            )
        )

@st.cache_resource(show_spinner=True)
def load_whisper(model_name: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, device=device, compute_type=compute_type)


apply_theme()
ensure_state()

env_device = _normalize_device_name(os.getenv("WHISPER_DEVICE"))
forced_device = _normalize_device_name(st.session_state.get("whisper_force_device"))
device = env_device or forced_device or _DEFAULT_DEVICE
compute_type = "int8_float16" if device == "cuda" else "int8"
model_name = WHISPER_MODEL_CUDA if device == "cuda" else WHISPER_MODEL_CPU

if WhisperModel is None:
    st.error('faster-whisper не установлен')
    st.stop()

whisper_meta: Dict[str, Any] = {
    "device": device,
    "compute_type": compute_type,
    "model": model_name,
    "fallback": False,
}

try:
    whisper_model = load_whisper(model_name, device, compute_type)
except Exception as exc:
    if device != "cpu":
        fallback_device = "cpu"
        fallback_compute_type = "int8"
        fallback_model_name = WHISPER_MODEL_CPU
        try:
            whisper_model = load_whisper(
                fallback_model_name, fallback_device, fallback_compute_type
            )
        except Exception as cpu_exc:
            st.error(
                "Не удалось загрузить модель Whisper: GPU-режим завершился ошибкой "
                f"({exc}). Попытка запуска на CPU также провалилась ({cpu_exc})."
            )
            st.stop()
        else:
            st.session_state["whisper_force_device"] = "cpu"
            st.warning(
                "Whisper переключён на CPU (int8) из-за ошибки при запуске на CUDA: "
                f"{exc}.\n"
                "Чтобы всегда использовать CPU, задайте переменную окружения WHISPER_DEVICE=cpu."
            )
            device = fallback_device
            compute_type = fallback_compute_type
            model_name = fallback_model_name
            whisper_meta = {
                "device": device,
                "compute_type": compute_type,
                "model": model_name,
                "fallback": True,
            }
    else:
        st.error(f'Не удалось загрузить модель Whisper: {exc}')
        st.stop()

st.session_state["whisper_meta"] = whisper_meta

st.title('Whisper → LLaMA → Jira')
render_metrics()

st.header('Загрузка и предпрослушивание')
uploaded = st.file_uploader(
    'Форматы: wav, mp3, m4a, ogg, flac, mp4, mov, mkv, webm',
    type=SUPPORTED_UPLOAD_TYPES,
)
if uploaded is not None:
    st.session_state['uploaded_file_name'] = uploaded.name
    st.session_state['uploaded_file_bytes'] = uploaded.getvalue()
if st.session_state.get('uploaded_file_bytes'):
    try:
        st.audio(st.session_state['uploaded_file_bytes'])
    except Exception:
        st.warning('Не удалось воспроизвести предпрослушивание')

st.header('Распознать (с выбором языка)')
col_btn, col_lang = st.columns([1, 1], gap='medium')
with col_btn:
    recognize_clicked = st.button('Распознать из аудио', type='primary')
with col_lang:
    lang_option = st.selectbox(
        'Язык',
        ['auto', 'ru', 'en', 'kk', 'tr'],
        index=['auto', 'ru', 'en', 'kk', 'tr'].index(st.session_state.get('lang_hint', 'auto')),
    )
    st.session_state['lang_hint'] = lang_option
if recognize_clicked:
    if not st.session_state.get('uploaded_file_bytes'):
        st.warning('Сначала загрузите файл')
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{st.session_state['uploaded_file_name']}")
        tmp.write(st.session_state['uploaded_file_bytes'])
        tmp.flush()
        tmp.close()
        src_path = tmp.name
        ext = (st.session_state['uploaded_file_name'].split('.')[-1] or '').lower()
        audio_path = src_path
        extract_ms = 0.0
        try:
            if ext in VIDEO_EXTENSIONS:
                start_extract = time.perf_counter()
                audio_path = extract_audio_to_wav(src_path)
                extract_ms = (time.perf_counter() - start_extract) * 1000
            else:
                extract_ms = 0.0
            st.session_state['metrics']['extract_audio_ms'] = extract_ms
            with st.spinner('Распознаю аудио...'):
                kwargs: Dict[str, Any] = {}
                lang_hint = st.session_state.get('lang_hint', 'auto')
                if lang_hint and lang_hint != 'auto':
                    kwargs['language'] = lang_hint
                start_asr = time.perf_counter()
                segments, _info = whisper_model.transcribe(
                    audio_path,
                    vad_filter=True,
                    vad_parameters={'min_silence_duration_ms': 500},
                    **kwargs,
                )
                parts = [segment.text for segment in segments]
                raw_text = ''.join(parts).strip()
                st.session_state['metrics']['asr_ms'] = (time.perf_counter() - start_asr) * 1000
                final_text = raw_text
                meta = st.session_state.get('llm_meta', {})
                if raw_text:
                    try:
                        start_clean = time.perf_counter()
                        cleaned_text, meta = llama_clean_text(raw_text)
                        st.session_state['metrics']['llm_clean_ms'] = (time.perf_counter() - start_clean) * 1000
                        final_text = cleaned_text.strip() or raw_text
                    except RuntimeError as exc:
                        st.session_state['metrics']['llm_clean_ms'] = None
                        if 'LLM endpoint not found' in str(exc):
                            st.error('LLM endpoint не найден. Проверьте переменные LLAMA_BASE или LLAMA_URL.')
                        else:
                            st.error(f'Ошибка LLM: {exc}')
                    except Exception as exc:
                        st.session_state['metrics']['llm_clean_ms'] = None
                        st.error(f'Ошибка очистки текста: {exc}')
                else:
                    st.session_state['metrics']['llm_clean_ms'] = 0.0
                st.session_state['transcript'] = final_text
                st.session_state['transcript_area'] = final_text
                st.session_state['llm_meta'] = meta
                st.success('Распознавание завершено')
        except RuntimeError as exc:
            st.session_state['metrics']['extract_audio_ms'] = None
            message = (
                'ffmpeg не найден. Установите ffmpeg и добавьте его в PATH.'
                if 'ffmpeg not found' in str(exc)
                else f'Не удалось извлечь аудио: {exc}'
            )
            st.error(message)
        except Exception as exc:
            st.session_state['metrics']['asr_ms'] = None
            st.error(f'Ошибка распознавания: {exc}')
        finally:
            try:
                os.unlink(src_path)
            except Exception:
                pass
            try:
                if audio_path != src_path:
                    os.unlink(audio_path)
            except Exception:
                pass
st.header('Распознанный текст')
placeholder_text = 'Тут появится распознанный и автоматически исправленный текст'
current_text = st.session_state.get('transcript_area') or st.session_state.get('transcript') or ''
st.text_area('Текст', value=current_text, placeholder=placeholder_text, height=280, key='transcript_area')

st.header('Извлечение задач')
extract_clicked = st.button('Извлечь задачи', type='secondary')
if extract_clicked:
    body = st.session_state.get('transcript_area') or st.session_state.get('transcript') or ''
    if not body.strip():
        st.warning('Нет текста для извлечения')
    else:
        with st.spinner('Извлекаю задачи...'):
            start_extract = time.perf_counter()
            try:
                tasks, meta = llama_extract_tasks(body)
                st.session_state['metrics']['extract_tasks_ms'] = (time.perf_counter() - start_extract) * 1000
                st.session_state['tasks'] = tasks
                st.session_state['llm_meta'] = meta
                st.session_state['metrics']['tasks_total'] = len(tasks)
                st.success(f'Извлечено задач: {len(tasks)}')
            except RuntimeError as exc:
                st.session_state['metrics']['extract_tasks_ms'] = None
                if 'LLM endpoint not found' in str(exc):
                    st.error('LLM endpoint не найден. Проверьте переменные LLAMA_BASE или LLAMA_URL.')
                else:
                    st.error(f'Ошибка LLM: {exc}')
            except Exception as exc:
                st.session_state['metrics']['extract_tasks_ms'] = None
                st.error(f'Не удалось извлечь задачи: {exc}')
st.header('Список задач')
tasks = st.session_state.get('tasks', [])
if tasks:
    normalize_clicked = st.button('Нормализовать дедлайны', type='secondary')
    if normalize_clicked:
        normalized = False
        for task in tasks:
            raw_due = str(task.get('due', '')).strip()
            if not raw_due:
                continue
            normalized_due = normalize_due_value(raw_due)
            if normalized_due and normalized_due != raw_due:
                task['due'] = normalized_due
                normalized = True
        if normalized:
            st.success('Дедлайны нормализованы')
        else:
            st.info('Не удалось распознать дедлайны')
    updated_tasks: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        task.setdefault('id', stable_id(8))
        with st.expander(f"Задача {index}: {task.get('summary') or ''}".strip()):
            summary = st.text_input('Тема', task.get('summary', ''), key=f"s_{task['id']}")
            description = st.text_area('Описание', task.get('description', ''), key=f"d_{task['id']}")
            labels = st.text_input('Метки (через запятую)', task.get('labels', ''), key=f"l_{task['id']}")
            due = st.text_input('Дата дедлайна', task.get('due', ''), key=f"due_{task['id']}")
            priority_value = task.get('priority', 'Medium')
            if priority_value not in PRIORITIES:
                priority_value = 'Medium'
            priority = st.selectbox(
                'Приоритет',
                PRIORITIES,
                index=PRIORITIES.index(priority_value),
                key=f"p_{task['id']}",
            )
            comment = st.text_area('Комментарий', task.get('comment', ''), key=f"c_{task['id']}")
            remove = st.button('Удалить', key=f"del_{task['id']}")
            if not remove:
                updated_tasks.append(
                    {
                        'id': task['id'],
                        'summary': summary,
                        'description': description,
                        'labels': labels,
                        'due': due,
                        'priority': priority,
                        'comment': comment,
                    }
                )
    st.session_state['tasks'] = updated_tasks
    st.session_state['metrics']['tasks_total'] = len(updated_tasks)
else:
    st.caption('Сначала извлеките задачи или добавьте их вручную ниже')

if st.button('Добавить задачу вручную'):
    st.session_state['tasks'].append(
        {
            'id': stable_id(8),
            'summary': '',
            'description': '',
            'labels': '',
            'due': '',
            'priority': 'Medium',
            'comment': '',
        }
    )
    st.session_state['metrics']['tasks_total'] = len(st.session_state['tasks'])
st.header('Отправка в Jira')
with st.form('jira_form', clear_on_submit=False):
    st.text_input('Jira URL', placeholder='https://your-domain.atlassian.net', key='jira_url')
    st.text_input('Jira Email', key='jira_email')
    st.text_input('Jira API Token', type='password', key='jira_token')
    st.text_input('Project Key', placeholder='PRJ', key='jira_project')
    submit_jira = st.form_submit_button('Создать задачи', type='primary')

st.markdown(
    """
    <script>
    (function(){
      function enhance(){
        const form = document.querySelector('form');
        if(!form){ setTimeout(enhance, 600); return; }
        const inputs = form.querySelectorAll('input, textarea');
        inputs.forEach((el, index) => {
          el.addEventListener('keydown', function(event){
            if(event.key === 'Enter' && !event.shiftKey){
              event.preventDefault();
              const next = inputs[index + 1];
              if(next){ next.focus(); }
              else {
                const button = Array.from(form.querySelectorAll('button')).find(btn => btn.innerText.trim().includes('Создать задачи'));
                if(button){ button.click(); }
              }
            }
          });
        });
      }
      setTimeout(enhance, 600);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)
if submit_jira:
    required = [
        ('URL', st.session_state.get('jira_url', '').strip()),
        ('Email', st.session_state.get('jira_email', '').strip()),
        ('API token', st.session_state.get('jira_token', '').strip()),
        ('Project Key', st.session_state.get('jira_project', '').strip()),
    ]
    missing = [name for name, value in required if not value]
    if missing:
        st.warning('Заполните поля: ' + ', '.join(missing))
    else:
        tasks_for_jira = list(st.session_state.get('tasks', []))
        if not tasks_for_jira:
            st.error('Нет задач для отправки')
        else:
            with st.spinner('Создаю задачи в Jira...'):
                start_jira = time.perf_counter()
                try:
                    created, errors, links = jira_create_all(
                        st.session_state['jira_url'],
                        st.session_state['jira_email'],
                        st.session_state['jira_token'],
                        st.session_state['jira_project'],
                        tasks_for_jira,
                    )
                    st.session_state['metrics']['jira_create_ms'] = (time.perf_counter() - start_jira) * 1000
                    st.session_state['metrics']['tasks_created'] = len(created)
                    st.session_state['metrics']['errors_count'] = len(errors)
                    if created:
                        st.success('Создано: ' + ', '.join(created))
                        st.write(
                            'Проект ' + st.session_state['jira_project'] + ': ' +
                            project_list_link(st.session_state['jira_url'], st.session_state['jira_project'])
                        )
                        for link in links:
                            st.write(link)
                    if errors:
                        st.error('Ошибки: ' + ' | '.join(error[:200] for error in errors if error))
                except RuntimeError as exc:
                    st.session_state['metrics']['jira_create_ms'] = None
                    st.error(f'Ошибка Jira: {exc}')
                except Exception as exc:
                    st.session_state['metrics']['jira_create_ms'] = None
                    st.error(f'Не удалось создать задачи: {exc}')
