
import os, re, json, uuid, shutil, tempfile
from datetime import datetime, timedelta, date, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests, streamlit as st

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = str(os.path.dirname(_FFMPEG)) + os.pathsep + os.environ.get("PATH","")
except Exception:
    _FFMPEG = shutil.which("ffmpeg")

try:
    import dateparser
except Exception:
    dateparser = None

st.set_page_config(page_title="Whisper → LLaMA → Jira", page_icon="🌀", layout="wide")

LLAMA_BASE = os.getenv("LLAMA_BASE", "https://vsjz8fv63q4oju-8000.proxy.runpod.net")
LLAMA_URL = os.getenv("LLAMA_URL", "")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "")
LLAMA_KEY = os.getenv("LLAMA_API_KEY", "app-V8dsYkzX4pu9544X7qUPrF4J")
LLAMA_AUTH_HEADER = os.getenv("LLAMA_AUTH_HEADER", "Authorization")
LLAMA_AUTH_SCHEME = os.getenv("LLAMA_AUTH_SCHEME", "Bearer")

DEVICE = "cuda" if os.system("nvidia-smi >/dev/null 2>&1") == 0 else "cpu"
WHISPER_SIZE = "medium"
COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"

SUPPORTED = ["wav","mp3","m4a","ogg","flac","mp4","mov","mkv","webm"]
PRIORITIES = ["Highest","High","Medium","Low","Lowest"]
MAX_SUMMARY = 160

def css():
    st.markdown(
        """
        <style>
        :root{
          --bg-deep:#0f0a23;
          --bg-card:rgba(255,255,255,0.06);
          --bg-input:#1b1233;
          --bg-inset:#241945;
          --bg-audio:#2a1d52;
          --bd-ac:#ff2e6a;
          --tx:#F2ECFF;
        }
        html,body,.stApp{background:var(--bg-deep)!important;color:var(--tx)!important}
        .block-container{max-width:1228px;padding-top:1.8rem;padding-bottom:2rem}
        h1,h2,h3,h4,h5,h6,label{color:var(--tx)!important;letter-spacing:.3px}
        [data-testid="stCaptionContainer"], .stCaption, small{color:var(--tx)!important;opacity:.92!important}
        header[data-testid="stHeader"], [data-testid="stToolbar"]{background:transparent!important}
        [data-testid="stAppViewContainer"]>.main{background:transparent!important}
        .stTextArea textarea,.stTextInput input,.stDateInput input{
            background:var(--bg-input)!important;color:var(--tx)!important;
            border:1.2px solid var(--bd-ac)!important;border-radius:16px!important;
            outline:none!important;box-shadow:none!important;
        }
        .stTextArea div, .stTextInput div{
            background:transparent!important;border:none!important;box-shadow:none!important;
        }
        [data-testid="stTextInput"] div[data-baseweb="input"]{
            background:var(--bg-input)!important;border:1.2px solid var(--bd-ac)!important;border-radius:16px!important;
        }
        [data-testid="stTextInput"] div[data-baseweb="input"] *{background:transparent!important}
        [data-testid="stTextInput"] div[data-testid="password-toggle"]{
            background:var(--bg-input)!important;border-left:1.2px solid var(--bd-ac)!important;
            border-top-right-radius:16px!important;border-bottom-right-radius:16px!important;
        }
        [data-testid="stTextInput"] div[data-testid="password-toggle"] button{
            background:transparent!important;color:var(--tx)!important;border:none!important;outline:none!important;
        }
        div[data-baseweb="select"]>div{
            background:var(--bg-input)!important;color:var(--tx)!important;
            border:1.2px solid var(--bd-ac)!important;border-radius:16px!important;
        }
        div[data-baseweb="select"] *{color:var(--tx)!important}
        [data-testid="stFileUploaderDropzone"]{
            background:var(--bg-inset)!important;color:var(--tx)!important;
            border:1.2px dashed var(--bd-ac)!important;border-radius:18px!important;
        }
        .uploadedFile, .uploadedFile *{background:transparent!important;color:var(--tx)!important}
        audio{width:100%;background:var(--bg-audio)!important;border:1.2px solid var(--bd-ac)!important;border-radius:22px!important}
        audio::-webkit-media-controls-panel{background:var(--bg-audio)!important}
        .stExpander{background:var(--bg-card)!important;border:1.2px solid var(--bd-ac)!important;border-radius:16px!important}
        .stAlert{background:var(--bg-card)!important;border:1.2px solid var(--bd-ac)!important;border-radius:14px!important;color:var(--tx)!important}
        .stButton button{
            border-radius:18px!important;padding:1rem 1.6rem;border:0!important;
            background:linear-gradient(90deg,#ff1e9b 0%,#7b2ff7 100%)!important;
            color:#0b121b!important;font-weight:800!important;letter-spacing:.2px;
            box-shadow:0 0 18px rgba(255,0,168,.45),0 0 28px rgba(115,0,255,.35)!important;
            white-space:nowrap!important;min-width:320px;line-height:1.15;outline:none!important;
        }
        .stButton button:hover{transform:translateY(-1px);box-shadow:0 0 22px rgba(255,0,168,.7),0 0 48px rgba(115,0,255,.55)!important}
        .stApp *[style*="background-color: #fff"], .stApp *[style*="rgb(255, 255, 255)"]{
            background:var(--bg-input)!important;color:var(--tx)!important;border-color:var(--bd-ac)!important;box-shadow:none!important;
        }
        .stApp *[style*="border-color: rgb(230, 234, 241)"]{border-color:var(--bd-ac)!important}
        .stSelectbox>div{border-radius:16px!important}
        .language-row{display:flex;align-items:center;gap:14px;margin-bottom:.6rem}
        .language-row .label{opacity:.92}
        </style>
        """,
        unsafe_allow_html=True,
    )

def ensure_state():
    st.session_state.setdefault("file_name","")
    st.session_state.setdefault("file_bytes",b"")
    st.session_state.setdefault("transcript","")
    st.session_state.setdefault("transcript_area","")
    st.session_state.setdefault("tasks",[])
    st.session_state.setdefault("lang","auto")
    st.session_state.setdefault("llama_mode","")
    st.session_state.setdefault("llama_url","")
    st.session_state.setdefault("llama_model","")

def sid(n:int=10)->str:
    return uuid.uuid4().hex[:n]

def to_iso(d:Any)->str:
    if not d: return ""
    if isinstance(d,date): return d.isoformat()
    s=str(d).strip()
    if re.match(r"^\\d{4}-\\d{2}-\\d{2}$",s): return s
    return ""

def parse_date_natural(s:str)->str:
    s=(s or "").strip().lower()
    if not s: return ""
    today=date.today()
    if s in ("сегодня","today"): return today.isoformat()
    if s in ("завтра","tomorrow"): return (today+timedelta(days=1)).isoformat()
    if s in ("послезавтра","day after tomorrow"): return (today+timedelta(days=2)).isoformat()
    m=re.match(r"^через\\s+(\\d+)\\s*(дн(я|ей|ь)?|day|days)$",s)
    if m: return (today+timedelta(days=int(m.group(1)))).isoformat()
    m=re.match(r"^через\\s+(\\d+)\\s*(недел(ю|и|ь|и)|week|weeks)$",s)
    if m: return (today+timedelta(days=int(m.group(1))*7)).isoformat()
    if dateparser:
        try:
            dt=dateparser.parse(s,languages=["ru","en"],settings={"PREFER_DATES_FROM":"future"})
            if dt: return dt.date().isoformat()
        except Exception: pass
    for pat in [r"^(\\d{2})[./-](\\d{2})[./-](\\d{4})$", r"^(\\d{4})[./-](\\d{2})[./-](\\d{2})$"]:
        m=re.match(pat,s)
        if m:
            try:
                if len(m.groups())==3 and len(m.group(1))==2:
                    return date(int(m.group(3)),int(m.group(2)),int(m.group(1))).isoformat()
                return date(int(m.group(1)),int(m.group(2)),int(m.group(3))).isoformat()
            except Exception: pass
    return ""

def ffmpeg_extract(src:str)->str:
    out=tempfile.NamedTemporaryFile(delete=False,suffix=".wav").name
    exe=_FFMPEG or shutil.which("ffmpeg")
    if not exe: raise RuntimeError("ffmpeg not found")
    cmd=f'"{exe}" -y -i "{src}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{out}"'
    if os.system(cmd)!=0 or not os.path.exists(out): raise RuntimeError("ffmpeg failed")
    return out

def llama_headers()->Dict[str,str]:
    h={"Content-Type":"application/json"}
    if LLAMA_KEY: h[LLAMA_AUTH_HEADER]=f"{LLAMA_AUTH_SCHEME} {LLAMA_KEY}"
    return h

def llama_models(base:str)->List[str]:
    try:
        r=requests.get(base.rstrip("/")+"/v1/models",headers=llama_headers(),timeout=30)
        if not r.ok: return []
        arr=r.json().get("data",[])
        out=[]
        for x in arr:
            mid=x.get("id") if isinstance(x,dict) else None
            if isinstance(mid,str): out.append(mid)
        return out
    except Exception:
        return []

def model_pick(models:List[str], prefer:str)->str:
    if prefer and prefer in models: return prefer
    if prefer:
        low=prefer.lower()
        for m in models:
            if m.lower()==low: return m
    ranked=[]
    for m in models:
        ml=m.lower();score=0
        if "instruct" in ml or "chat" in ml: score+=3
        if "llama" in ml: score+=2
        if "scout" in ml: score+=1
        if "fp8" in ml: score+=1
        ranked.append((score,m))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    return ranked[0][1] if ranked else ""

def try_mode(base:str, model:str)->Tuple[str,str]:
    urlc=base.rstrip("/")+"/v1/chat/completions"
    urlr=base.rstrip("/")+"/v1/responses"
    m={"model":model or "llama","temperature":0.1}
    try:
        rc=requests.post(urlc,headers=llama_headers(),json={**m,"messages":[{"role":"user","content":"ping"}]},timeout=25)
        if rc.status_code==200: return "chat",urlc
    except Exception: pass
    try:
        rr=requests.post(urlr,headers=llama_headers(),json={**m,"input":[{"role":"user","content":"ping"}]},timeout=25)
        if rr.status_code==200: return "responses",urlr
    except Exception: pass
    return "",""

def autodiscover()->Tuple[str,str,str]:
    base=LLAMA_BASE.strip().rstrip("/")
    if LLAMA_URL.strip():
        u=LLAMA_URL.strip()
        mode="chat" if "/chat/completions" in u else ("responses" if "/responses" in u else "")
        return mode,u,LLAMA_MODEL or ""
    models=llama_models(base)
    model=model_pick(models,LLAMA_MODEL)
    mode,url=try_mode(base, model or (models[0] if models else "llama"))
    return mode,url,model or (models[0] if models else "llama")

def llama_call(mode:str,url:str,model:str,msgs:List[Dict[str,str]])->str:
    if mode=="chat":
        payload={"model":model,"messages":msgs,"temperature":0.1,"max_tokens":4000}
    else:
        payload={"model":model,"input":msgs,"temperature":0.1,"max_tokens":4000}
    r=requests.post(url,headers=llama_headers(),json=payload,timeout=180)
    r.raise_for_status()
    data=r.json()
    if mode=="chat":
        return data["choices"][0]["message"]["content"]
    txt=data.get("output_text")
    if isinstance(txt,str) and txt.strip(): return txt
    out=data.get("output",[]) or data.get("choices",[])
    if out and isinstance(out,list):
        first=out[0]
        return first.get("content") or first.get("message",{}).get("content") or ""
    return ""

def llama_clean(s:str)->Tuple[str,Dict[str,str]]:
    mode,url,model=autodiscover()
    if not mode or not url: raise RuntimeError("LLM endpoint not found")
    sys="Ты редактор текста. Исправь опечатки и пунктуацию, не меняй смысл. Верни только исправленный текст."
    out=llama_call(mode,url,model,[{"role":"system","content":sys},{"role":"user","content":s}]).strip()
    return out or s, {"mode":mode,"url":url,"model":model}

def autolabels_from_summary(s:str)->str:
    w=[x.lower() for x in re.findall(r"[\\w\\-А-Яа-яЁё]{3,}", s)]
    seen=set(); out=[]
    for x in w:
        if x in seen: continue
        seen.add(x); out.append(x)
        if len(out)>=4: break
    return ", ".join(out)

def parse_tasks_json(txt:str)->List[Dict[str,Any]]:
    m=re.search(r"\\[[\\s\\S]*\\]",txt)
    blob=m.group(0) if m else txt
    data=json.loads(blob)
    if not isinstance(data,list): raise ValueError("not list")
    out=[]
    for it in data:
        if not isinstance(it,dict): continue
        summary=str(it.get("summary","")).strip()[:MAX_SUMMARY]
        desc=str(it.get("description","")).strip()
        labels_raw=str(it.get("labels","")).strip()
        if not labels_raw and summary: labels_raw=autolabels_from_summary(summary)
        parts=[p.strip() for p in labels_raw.split(",") if p.strip()]
        due_raw=str(it.get("due","")).strip()
        due_iso=parse_date_natural(due_raw) or to_iso(due_raw) if due_raw else ""
        comment=str(it.get("comment","")).strip()
        pr=str(it.get("priority","") or "Medium").title()
        if pr not in PRIORITIES: pr="Medium"
        out.append({"id":uuid.uuid4().hex[:8],"summary":summary,"description":desc,"labels":", ".join(parts),"due":due_iso,"comment":comment,"priority":pr})
    return out

def llama_extract(transcript:str)->Tuple[List[Dict[str,Any]],Dict[str,str]]:
    mode,url,model=autodiscover()
    if not mode or not url: raise RuntimeError("LLM endpoint not found")
    sys=(
        "Извлеки задачи из текста. Верни ТОЛЬКО JSON-массив. "
        "Элемент: {summary, description, labels, due, comment, priority}. "
        "labels — 3-6 ключевых слов через запятую. "
        "priority одно из Highest, High, Medium, Low, Lowest. "
        "Если в тексте есть фразы «завтра», «послезавтра», «через N дней/недели» — заполни due такой фразой."
    )
    txt=llama_call(mode,url,model,[{"role":"system","content":sys},{"role":"user","content":transcript}])
    tasks=parse_tasks_json(txt)
    return tasks, {"mode":mode,"url":url,"model":model}

def jira_priority_id(base:str,email:str,token:str,name:str)->Optional[str]:
    try:
        r=requests.get(base.rstrip("/")+"/rest/api/3/priority",auth=(email,token),timeout=40)
        if r.status_code>=300: return None
        for it in r.json():
            if str(it.get("name","")).lower()==name.lower():
                return it.get("id")
    except Exception: return None
    return None

def jira_create_issue(base:str,email:str,token:str,project:str,t:Dict[str,Any])->Dict[str,Any]:
    url=base.rstrip("/")+"/rest/api/3/issue"
    hdr={"Accept":"application/json","Content-Type":"application/json"}
    fields={"project":{"key":project},"summary":(t.get("summary") or "Задача")[:MAX_SUMMARY],"issuetype":{"name":"Task"}}
    raw=t.get("labels","") or ""
    labels=[x.strip() for x in raw.split(",") if x.strip()]
    if labels: fields["labels"]=labels
    if t.get("due"):
        iso = to_iso(t.get("due")) or parse_date_natural(t.get("due"))
        if iso: fields["duedate"]=iso
    pr=t.get("priority") or "Medium"
    pid=jira_priority_id(base,email,token,pr)
    fields["priority"]={"id":pid} if pid else {"name":pr}
    desc=str(t.get("description","")).strip()
    if desc: fields["description"]={"type":"doc","version":1,"content":[{"type":"paragraph","content":[{"type":"text","text":desc}]}]}
    body={"fields":fields}
    r=requests.post(url,auth=(email,token),headers=hdr,json=body,timeout=60)
    if r.status_code>=300: return {"ok":False,"error":r.text}
    return {"ok":True,**r.json()}

def jira_comment(base:str,email:str,token:str,key:str,text:str)->Dict[str,Any]:
    if not (text or "").strip(): return {"ok":True,"skipped":True}
    url=base.rstrip("/")+f"/rest/api/3/issue/{key}/comment"
    hdr={"Accept":"application/json","Content-Type":"application/json"}
    payload={"body":{"type":"doc","version":1,"content":[{"type":"paragraph","content":[{"type":"text","text":text}]}]}}
    r=requests.post(url,auth=(email,token),headers=hdr,json=payload,timeout=60)
    if r.status_code>=300: return {"ok":False,"error":r.text}
    return {"ok":True,**r.json()}

def issue_link(base:str,key:str)->str:
    return base.rstrip("/")+"/browse/"+key

def project_list(base:str,key:str)->str:
    return base.rstrip("/")+f"/jira/core/projects/{key}/list"

css()
def ensure_state():
    st.session_state.setdefault("file_name","")
    st.session_state.setdefault("file_bytes",b"")
    st.session_state.setdefault("transcript","")
    st.session_state.setdefault("transcript_area","")
    st.session_state.setdefault("tasks",[])
    st.session_state.setdefault("lang","auto")
    st.session_state.setdefault("llama_mode","")
    st.session_state.setdefault("llama_url","")
    st.session_state.setdefault("llama_model","")
ensure_state()

@st.cache_resource(show_spinner=True)
def load_whisper()->WhisperModel:
    return WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

if WhisperModel is None:
    st.error("faster-whisper не установлен")
    st.stop()
whisper=load_whisper()

st.title("Whisper → LLaMA → Jira")

st.header("Загрузка и распознавание")
up=st.file_uploader("Форматы: wav, mp3, m4a, ogg, flac, mp4, mov, mkv, webm", type=SUPPORTED)
if up is not None:
    st.session_state["file_name"]=up.name
    st.session_state["file_bytes"]=up.getvalue()
if st.session_state.get("file_bytes"):
    st.audio(st.session_state["file_bytes"])

st.header("Распознать")
col = st.container()
with col:
    st.markdown('<div class="language-row"><div class="label">Язык</div>', unsafe_allow_html=True)
    lang = st.selectbox("", ["auto","ru","en","kk","tr"], index=["auto","ru","en","kk","tr"].index(st.session_state.get("lang","auto")), label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["lang"]=lang
    go_rec=st.button("Распознать из аудио", type="primary", use_container_width=False)

if go_rec:
    if not st.session_state.get("file_bytes"):
        st.warning("Сначала загрузите файл")
    else:
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=f"_{st.session_state['file_name']}")
        tmp.write(st.session_state["file_bytes"]); tmp.flush(); tmp.close()
        src=tmp.name
        ext=(st.session_state["file_name"].split(".")[-1] or "").lower()
        wav_path=src
        if ext in ["mp4","mov","mkv","webm"]: wav_path=ffmpeg_extract(src)
        try:
            st.info("Распознаю, подождите...")
            kw={}; lang=st.session_state.get("lang","auto")
            if lang and lang!="auto": kw["language"]=lang
            parts=[]
            segs, info = whisper.transcribe(wav_path, vad_filter=True, vad_parameters={"min_silence_duration_ms":500}, **kw)
            for s in segs: parts.append(s.text)
            raw="".join(parts).strip()
            cleaned,meta = llama_clean(raw) if raw else ("",{})
            final=cleaned or raw
            st.session_state["transcript"]=final
            st.session_state["transcript_area"]=final
            st.session_state["llama_mode"]=meta.get("mode","")
            st.session_state["llama_url"]=meta.get("url","")
            st.session_state["llama_model"]=meta.get("model","")
            st.success("Готово")
        finally:
            try: os.unlink(src)
            except Exception: pass
            try:
                if wav_path!=src: os.unlink(wav_path)
            except Exception: pass

st.header("Распознанный текст")
placeholder="Тут появится распознанный и автоматически исправленный текст"
current=st.session_state.get("transcript_area") or st.session_state.get("transcript") or ""
edited=st.text_area("Текст", value=current, placeholder=placeholder, height=260, key="transcript_area")

go_ext=st.button("Извлечь задачи", type="secondary")
if go_ext:
    body=edited or st.session_state.get("transcript","")
    if not body.strip():
        st.warning("Нет текста для извлечения")
    else:
        try:
            tasks,meta=llama_extract(body)
            st.session_state["tasks"]=tasks
            st.session_state["llama_mode"]=meta.get("mode","")
            st.session_state["llama_url"]=meta.get("url","")
            st.session_state["llama_model"]=meta.get("model","")
            st.success(f"Извлечено задач: {len(tasks)}")
        except Exception as e:
            st.error(str(e))

st.header("Список задач")
new=[]
if st.session_state.get("tasks"):
    idx=1
    for t in st.session_state.get("tasks",[]):
        with st.expander(f"Задача {idx}: {t.get('summary') or ''}", expanded=False):
            t["summary"]=st.text_input("Тема", t.get("summary",""), key=f"s_{t['id']}")
            t["description"]=st.text_area("Описание", t.get("description",""), key=f"d_{t['id']}")
            t["labels"]=st.text_input("Метки (через запятую)", t.get("labels",""), key=f"l_{t['id']}")
            t["due"]=st.text_input("Дата дедлайна", value=t.get("due",""), placeholder="YYYY-MM-DD пример: 2025-09-21", key=f"due_{t['id']}")
            pr_default=t.get("priority","Medium")
            try: pr_idx=PRIORITIES.index(pr_default) if pr_default in PRIORITIES else PRIORITIES.index("Medium")
            except Exception: pr_idx=PRIORITIES.index("Medium")
            t["priority"]=st.selectbox("Приоритет", PRIORITIES, index=pr_idx, key=f"p_{t['id']}")
            t["comment"]=st.text_area("Комментарий", t.get("comment",""), key=f"c_{t['id']}")
            if st.button("Удалить", key=f"del_{t['id']}"):
                pass
            else:
                new.append(t)
        idx+=1
st.session_state["tasks"]=new

if st.button("Добавить задачу вручную"):
    st.session_state["tasks"].append({"id":uuid.uuid4().hex[:8],"summary":"","description":"","labels":"","due":"","comment":"","priority":"Medium"})

st.header("Отправка в Jira")
with st.form("jira_form"):
    jira_url=st.text_input("Jira URL", placeholder="https://your-domain.atlassian.net", key="jira_url")
    jira_email=st.text_input("Jira Email", key="jira_email")
    jira_token=st.text_input("Jira API Token", type="password", key="jira_token")
    jira_project=st.text_input("Project Key", placeholder="PRJ", key="jira_project")
    submit=st.form_submit_button("Создать задачи", type="primary")

st.markdown(
    """
    <script>
    (function(){
      function enh(){
        const frm=document.querySelector('form');
        if(!frm){ setTimeout(enh,700); return; }
        const inputs=frm.querySelectorAll('input, textarea');
        inputs.forEach((el,idx)=>{
          el.addEventListener('keydown',e=>{
            if(e.key==='Enter' && !e.shiftKey){
              e.preventDefault();
              const nx=inputs[idx+1];
              if(nx){ nx.focus(); } else {
                const btn=[...frm.querySelectorAll('button')].find(b=>b.innerText.trim().includes('Создать задачи'));
                if(btn) btn.click();
              }
            }
          });
        });
      }
      setTimeout(enh,800);
    })();
    </script>
    """, unsafe_allow_html=True
)

def jira_bulk_create():
    base=st.session_state.get("jira_url","").strip()
    em=st.session_state.get("jira_email","").strip()
    tok=st.session_state.get("jira_token","").strip()
    proj=st.session_state.get("jira_project","").strip()
    need=[("URL",base),("Email",em),("API token",tok),("Project Key",proj)]
    miss=[x for x,v in need if not v]
    if miss:
        st.warning("Заполните поля: "+", ".join(miss)); return
    tlist=list(st.session_state.get("tasks",[]))
    if not tlist:
        st.error("Нет задач для отправки"); return
    ok=[]; err=[]; links=[]
    for t in tlist:
        if t.get("due") and not re.match(r"^\\d{4}-\\d{2}-\\d{2}$", t["due"]):
            iso=parse_date_natural(t["due"])
            if iso: t["due"]=iso
        res=jira_create_issue(base,em,tok,proj,t)
        if not res.get("ok"):
            err.append(res.get("error","")); continue
        key=res.get("key") or res.get("id") or "?"
        if (t.get("comment") or "").strip():
            _c=jira_comment(base,em,tok,key,t["comment"].strip())
            if not _c.get("ok"): err.append(_c.get("error",""))
        ok.append(key); links.append(base.rstrip('/')+'/browse/'+key)
    if ok:
        st.success("Создано: "+", ".join(ok))
        st.write("Проект "+proj+": "+(base.rstrip('/')+f"/jira/core/projects/{proj}/list"))
        for u in links: st.write(u)
    if err: st.error("Ошибки: "+" | ".join([e[:200] for e in err]))

if submit:
    jira_bulk_create()
