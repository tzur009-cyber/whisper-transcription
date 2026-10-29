import atexit
import hashlib
import os
import gc
import json
import sqlite3
from datetime import datetime
import html
import time
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import deque
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from faster_whisper import WhisperModel


# Files remain available until the app exits.
WORK_DIR = Path(tempfile.mkdtemp(prefix="whisper_queue_"))
atexit.register(lambda: shutil.rmtree(WORK_DIR, ignore_errors=True))

LOCK = threading.RLock()
JOBS = {}
PENDING = deque()
ACTIVE = None
WORKER_RUNNING = False
STOP_CURRENT = threading.Event()
MAX_PENDING = 40
DATA_DIR = Path.home() / ".local" / "share" / "whisper-transcriber"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.chmod(0o700)
DB_PATH = DATA_DIR / "history.sqlite3"
DRIVE_REMOTE = "gdrive"
DRIVE_FOLDER = "התמלולים שלי"
DB_LOCK = threading.RLock()
SESSION_UPLOADS = set()


def init_database():
    with DB_LOCK, sqlite3.connect(DB_PATH) as db:
        db.execute("CREATE TABLE IF NOT EXISTS recordings (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.commit()
    DB_PATH.chmod(0o600)


def save_job(job):
    snapshot = dict(job)
    # Uploaded audio remains temporary. Only transcripts and metadata are persistent.
    snapshot["path"] = None
    snapshot["upload_source"] = None
    snapshot["started_at"] = None
    with DB_LOCK, sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT OR REPLACE INTO recordings (id, payload) VALUES (?, ?)",
            (snapshot["id"], json.dumps(snapshot, ensure_ascii=False)),
        )
        db.commit()


def load_history():
    with DB_LOCK, sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT payload FROM recordings ORDER BY rowid").fetchall()
    for (payload,) in rows:
        try:
            job = json.loads(payload)
            job["path"] = None
            job["upload_source"] = None
            job["started_at"] = None
            job.setdefault("drive_status", "טרם סונכרן")
            job.setdefault("created_at", "")
            if job.get("state") not in ("הושלם", "נעצר", "שגיאה", "נקטע בהפעלה קודמת"):
                job["state"] = "נקטע בהפעלה קודמת"
                job["error"] = "התמלול נקטע כשהשרת נסגר. הטקסט שנוצר נשמר."
                save_job(job)
            JOBS[job["id"]] = job
        except Exception as exc:
            print("Could not load one history entry:", exc)


def drive_configured():
    if not shutil.which("rclone"):
        return False
    try:
        result = subprocess.run(
            ["rclone", "listremotes"], capture_output=True, text=True, timeout=12
        )
        return result.returncode == 0 and f"{DRIVE_REMOTE}:" in result.stdout.splitlines()
    except Exception:
        return False


def drive_upload(job):
    if not drive_configured():
        return "ממתין לחיבור Google Drive"
    text = job.get("text", "").strip()
    if not text:
        return "אין טקסט להעלאה"
    name = Path(job.get("name", "הקלטה")).stem
    safe_name = re.sub(r'[\/\x00-\x1f<>:"|?*]', "_", name).strip(" .")[:75] or "הקלטה"
    stamp = job.get("created_at", "").replace(":", "-").replace(" ", "_")
    target = f"{DRIVE_REMOTE}:{DRIVE_FOLDER}/{safe_name}_{stamp}_{job['id'][:8]}.txt"
    body = f"שם הקלטה: {job.get('name', '')}\nתאריך: {job.get('created_at', '')}\nמודל: {job.get('model', '')}\n\n{text}"
    if job.get("comparison"):
        body += "\n\n--- תמלול מודל נוסף ---\n\n" + job["comparison"]
    try:
        result = subprocess.run(
            ["rclone", "rcat", target],
            input=body.encode("utf-8"), capture_output=True, timeout=180,
        )
        if result.returncode == 0:
            return "נשמר ב־Google Drive"
        error = result.stderr.decode("utf-8", errors="replace")[-220:]
        return "סנכרון נכשל: " + error
    except Exception as exc:
        return "סנכרון נכשל: " + str(exc)


def sync_all():
    if not drive_configured():
        return "Google Drive עדיין לא מחובר. התקיני rclone והגדירי חיבור בשם gdrive."
    with LOCK:
        candidates = [dict(j) for j in JOBS.values()
                      if j.get("state") == "הושלם" and j.get("text")
                      and j.get("drive_status") != "נשמר ב־Google Drive"]
    success = 0
    for job in candidates:
        status = drive_upload(job)
        with LOCK:
            if job["id"] in JOBS:
                JOBS[job["id"]]["drive_status"] = status
                save_job(JOBS[job["id"]])
        if status == "נשמר ב־Google Drive":
            success += 1
    return f"סונכרנו {success} מתוך {len(candidates)} תמלולים."


def check_drive():
    if not drive_configured():
        return "⚠ Google Drive לא מוגדר. לאחר הגדרת rclone בשם gdrive, לחצי שוב לבדיקה."
    try:
        result = subprocess.run(
            ["rclone", "lsf", f"{DRIVE_REMOTE}:"],
            capture_output=True, text=True, timeout=35,
        )
        if result.returncode == 0:
            return "✅ החיבור ל־Google Drive תקין. התמלולים יועלו אוטומטית בסיום."
        return "⚠ לא ניתן לגשת ל־Google Drive: " + result.stderr[-250:]
    except Exception as exc:
        return "⚠ בדיקת החיבור נכשלה: " + str(exc)


init_database()
load_history()

CSS = """
/* Bright orange hover outline around the file uploader. */
#audio-upload {
    border: 2px solid transparent !important;
    border-radius: 12px !important;
    transition: border-color .2s ease, box-shadow .2s ease !important;
}
#audio-upload:hover, #audio-upload:focus-within {
    border-color: #ff7a00 !important;
    box-shadow: 0 0 0 2px rgba(255,122,0,.30),
                0 0 22px rgba(255,111,0,.65) !important;
}
#transcribe-button:hover {
    box-shadow: 0 0 0 2px rgba(255,122,0,.40),
                0 0 18px rgba(255,122,0,.45) !important;
}
.whisper-progress {
    direction: rtl;
    text-align: right;
    padding: 13px 15px;
    background: #29292d;
    border: 1px solid #45454c;
    border-radius: 12px;
    margin: 8px 0;
}
.whisper-progress-top {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 10px;
    font-size: 14px;
}
.whisper-progress-track {
    direction: ltr;
    background: #55565b;
    border-radius: 999px;
    height: 13px;
    overflow: hidden;
}
.whisper-progress-fill {
    background: linear-gradient(90deg, #ff8a22, #ff5c00);
    border-radius: 999px;
    height: 100%;
    transition: width .45s ease;
}
.whisper-progress-meta {
    margin-top: 8px;
    color: #c4c4cb;
    font-size: 12px;
}
.gradio-container {
    direction: rtl;
    text-align: right;
    width: 94vw !important;
    max-width: 1380px !important;
    margin: auto !important;
}
textarea {
    direction: rtl !important;
    text-align: right !important;
    line-height: 1.85 !important;
    font-size: 17px !important;
}
"""


@lru_cache(maxsize=1)
def get_model(name):
    return WhisperModel(
        name,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )


def progress_card(job, empty_message="עדיין לא התחיל תמלול"):
    if not job:
        return '<div class="whisper-progress">' + html.escape(empty_message) + '</div>'

    percent = max(0, min(100, int(job.get("percent", 0))))
    name = html.escape(job.get("name", "הקלטה"))
    state = html.escape(job.get("state", "ממתין"))
    started = job.get("started_at")
    elapsed = int(time.monotonic() - started) if started is not None else int(job.get("elapsed_seconds") or 0)
    mins, secs = divmod(max(0, elapsed), 60)
    duration = f"{mins:02d}:{secs:02d}"
    finished = job.get("state") in ("הושלם", "נעצר", "שגיאה")
    time_label = "זמן תמלול סופי" if finished else "זמן שעבר"
    audio_seconds = job.get("audio_duration_seconds")
    speed = (f" · מהירות פי {float(audio_seconds) / elapsed:.1f} מזמן אמת"
             if finished and audio_seconds and elapsed > 0 and job.get("state") == "הושלם" else "")
    return (
        '<div class="whisper-progress">'
        f'<div class="whisper-progress-top"><strong>{name}</strong>'
        f'<span>{state} · <b>{percent}%</b></span></div>'
        f'<div class="whisper-progress-track" role="progressbar" '
        f'aria-label="התקדמות התמלול" aria-valuenow="{percent}" '
        f'aria-valuemin="0" aria-valuemax="100">'
        f'<div class="whisper-progress-fill" style="width:{percent}%"></div></div>'
        f'<div class="whisper-progress-meta">{time_label}: {duration}{speed} · '
        'ההתקדמות משוערת לפי משך השמע שעובד</div></div>'
    )


def make_paragraphs(parts):
    result = []
    current = []
    count = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        current.append(part)
        count += len(part.split())
        if count >= 65 and re.search(r'[.!?…]["״\']?$', part):
            result.append(" ".join(current))
            current = []
            count = 0

    if current:
        result.append(" ".join(current))

    return "\n\n".join(result)


def clean_audio(path, destination):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg אינו מותקן")

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(path), "-vn",
        "-af", "highpass=f=90,lowpass=f=7400,afftdn=nf=-25",
        "-ar", "16000", "-ac", "1", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ניקוי הרעשים נכשל: " + result.stderr[-300:])
    return str(destination)


# WHISPER_SPEAKERS_V1
# Voice embeddings are computed locally. Audio and embeddings are never sent to a service.
# The original transcript stays available if diarization fails or is disabled.

def speaker_turns(job):
    turns = []
    for row in job.get('speaker_segments') or []:
        if turns and turns[-1]['speaker'] == row['speaker'] and float(row['start']) - float(turns[-1]['end']) < 2.5:
            turns[-1]['text'] += ' ' + str(row['text'])
            turns[-1]['end'] = row['end']
        else:
            turns.append(dict(row))
    return turns


def speaker_plain_text(job):
    segments = speaker_turns(job)
    if not segments:
        return str(job.get("text", ""))
    names = job.get("speaker_names") or {}
    return "\n\n".join(
        f"{names.get(str(row['speaker']), 'דובר ' + str(row['speaker']))}:\n{row['text']}"
        for row in segments
    )


def speaker_reader_html(job):
    segments = speaker_turns(job)
    if not segments:
        return reader_text_html(job.get("text", ""))
    names = job.get("speaker_names") or {}
    chunks = []
    for row in segments:
        name = names.get(str(row["speaker"]), "דובר " + str(row["speaker"]))
        chunks.append('<div class="speaker-turn"><strong>' + html.escape(name) +
                      ':</strong><div>' + html.escape(row["text"]) + '</div></div>')
    return '<article class="reader-text-content" dir="rtl">' + "".join(chunks) + '</article>'


def speaker_embeddings(audio_path, rows):
    # Decode once to a temporary 16-kHz mono PCM WAV; large recordings are memory-mapped.
    import numpy as np
    from scipy.io import wavfile
    from resemblyzer import VoiceEncoder, preprocess_wav
    target = WORK_DIR / (uuid.uuid4().hex + '-speakers.wav')
    try:
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-i', str(audio_path), '-vn', '-ac', '1', '-ar', '16000',
               '-c:a', 'pcm_s16le', str(target)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('פענוח השמע נכשל: ' + result.stderr[-180:])
        sr, samples = wavfile.read(str(target), mmap=True)
        encoder = VoiceEncoder(device='cpu')
        duration = len(samples) / sr
        embeddings, indexes = [], []
        for i, row in enumerate(rows):
            start = max(0., float(row['start']) - .15)
            end = min(duration, float(row['end']) + .15)
            if end - start < .9:
                center = (start + end) / 2
                start = max(0., center - .6)
                end = min(duration, center + .6)
            if end - start < .6:
                continue
            clip = np.asarray(samples[int(start * sr):int(end * sr)], dtype=np.float32) / 32768.0
            try:
                clip = preprocess_wav(clip, source_sr=sr)
                if len(clip) < int(.45 * sr):
                    continue
                vector = encoder.embed_utterance(clip)
                embeddings.append(vector)
                indexes.append(i)
            except (ValueError, RuntimeError):
                continue
        if not embeddings:
            raise RuntimeError('לא נמצאו מספיק מקטעי דיבור לזיהוי דוברים.')
        return np.asarray(embeddings), indexes
    finally:
        target.unlink(missing_ok=True)


def speaker_cluster(vectors, mode, requested):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    size = len(vectors)
    if mode == 'קבוע':
        k = int(requested)
        if k > size:
            raise RuntimeError(f'נמצאו רק {size} מקטעים מתאימים. נסי פחות דוברים.')
    else:
        k = 1
        best = .24
        for count in range(2, min(8, size - 1) + 1):
            try:
                labels = AgglomerativeClustering(n_clusters=count, metric='cosine', linkage='average').fit_predict(vectors)
                score = float(silhouette_score(vectors, labels, metric='cosine'))
                # Require meaningful evidence before splitting into more voices.
                if score > best + .025 * (count - 2):
                    best, k = score, count
            except (ValueError, TypeError):
                continue
    if k == 1:
        return np.zeros(size, dtype=int)
    return AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(vectors)


def diarize_speech(audio_path, speech_rows, mode='אוטומטי', requested=2):
    import numpy as np
    if not speech_rows:
        raise RuntimeError('אין מקטעי דיבור לזיהוי דוברים.')
    vectors, indexes = speaker_embeddings(audio_path, speech_rows)
    labels = speaker_cluster(vectors, mode, requested)
    # Cluster IDs are assigned in order of first appearance, not by model internals.
    order, assigned = {}, {}
    for index, label in zip(indexes, labels):
        label = int(label)
        if label not in order:
            order[label] = len(order)
        assigned[index] = order[label]
    # Very short utterances inherit the nearest reliably classified utterance.
    for index in range(len(speech_rows)):
        if index not in assigned:
            closest = min(indexes, key=lambda j: abs(j - index))
            assigned[index] = assigned[closest]
    rows = [dict(start=float(r['start']), end=float(r['end']),
                 text=str(r['text']), speaker=assigned[i])
            for i, r in enumerate(speech_rows)]
    return rows, {str(i): f'דובר {i}' for i in sorted(set(assigned.values()))}


def speaker_controls(job_id):
    with LOCK:
        job = dict(JOBS.get(job_id) or {})
    rows = job.get('speaker_segments') or []
    if not rows:
        info = job.get('speaker_error') or 'אין עדיין הפרדת דוברים בתמלול הזה.'
        return (gr.update(visible=bool(job_id)),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(value=''),
                gr.update(value=str(job.get('speaker_count') or 2)),
                info)
    names = job.get('speaker_names') or {}
    speaker_ids = sorted({int(r['speaker']) for r in rows})
    choices = [(names.get(str(i), f'דובר {i}'), str(i)) for i in speaker_ids]
    segment_choices = [(f"{n + 1}. {str(r['text'])[:65]}", str(n)) for n, r in enumerate(rows)]
    return (gr.update(visible=True),
            gr.update(choices=segment_choices, value='0'),
            gr.update(choices=choices, value=str(rows[0]['speaker'])),
            gr.update(choices=choices, value=str(speaker_ids[0])),
            gr.update(choices=choices, value=str(speaker_ids[-1])),
            gr.update(value=rows[0]['text']),
            gr.update(value=str(len(speaker_ids))),
            job.get('speaker_error') or f'זוהו {len(speaker_ids)} דוברים. ניתן לתקן כל מקטע.')


def speaker_pick_segment(job_id, index):
    with LOCK:
        rows = (JOBS.get(job_id) or {}).get('speaker_segments') or []
        try:
            row = rows[int(index)]
            return str(row['speaker']), str(row['text'])
        except (TypeError, ValueError, IndexError, KeyError):
            return gr.update(), ''


def speaker_save_segment(job_id, index, speaker, text_value):
    text_value = str(text_value or '').strip()
    with LOCK:
        job = JOBS.get(job_id)
        if not job or not job.get('speaker_segments'):
            return gr.update(), gr.update(), 'אין מקטעים לעריכה.'
        if job_id == ACTIVE or job_id in PENDING:
            return gr.update(), gr.update(), 'לא ניתן לערוך בזמן תמלול.'
        try:
            row = job['speaker_segments'][int(index)]
            sid = int(speaker)
        except (ValueError, TypeError, IndexError):
            return gr.update(), gr.update(), 'בחרי מקטע ודובר.'
        if not text_value or sid not in {int(r['speaker']) for r in job['speaker_segments']}:
            return gr.update(), gr.update(), 'יש להזין טקסט ולבחור דובר קיים.'
        row['speaker'], row['text'] = sid, text_value
        job['text'] = speaker_plain_text(job)
        save_job(job)
        return speaker_reader_html(job), job['text'], '✅ תיקון המקטע נשמר במחשב.'


def speaker_rename(job_id, speaker, new_name):
    name = str(new_name or '').strip()
    if not name or len(name) > 60 or any(c in name for c in '\r\n<>'):
        return gr.update(), gr.update(), 'יש להזין שם תקין (עד 60 תווים).'
    with LOCK:
        job = JOBS.get(job_id)
        if not job or not job.get('speaker_segments'):
            return gr.update(), gr.update(), 'אין דוברים לשינוי.'
        if job_id == ACTIVE or job_id in PENDING:
            return gr.update(), gr.update(), 'לא ניתן לערוך בזמן תמלול.'
        job.setdefault('speaker_names', {})[str(speaker)] = name
        job['text'] = speaker_plain_text(job)
        save_job(job)
        return speaker_reader_html(job), job['text'], '✅ שם הדובר נשמר.'


def speaker_merge(job_id, from_id, into_id):
    if from_id == into_id or from_id is None or into_id is None:
        return gr.update(), gr.update(), 'בחרי שני דוברים שונים.'
    with LOCK:
        job = JOBS.get(job_id)
        if not job or not job.get('speaker_segments'):
            return gr.update(), gr.update(), 'אין דוברים לאיחוד.'
        if job_id == ACTIVE or job_id in PENDING:
            return gr.update(), gr.update(), 'לא ניתן לערוך בזמן תמלול.'
        for row in job['speaker_segments']:
            if str(row['speaker']) == str(from_id):
                row['speaker'] = int(into_id)
        job['text'] = speaker_plain_text(job)
        save_job(job)
        return speaker_reader_html(job), job['text'], '✅ הדוברים אוחדו. רענני את התמלול כדי לעדכן את הרשימות.'


def speaker_reanalyze(job_id, mode, count):
    with LOCK:
        job = JOBS.get(job_id)
        if not job or not job.get('speech_segments'):
            return gr.update(), gr.update(), 'לניתוח מחדש נדרשים מקטעים עם חותמות זמן. האפשרות זמינה לתמלולים חדשים.'
        if job_id == ACTIVE or job_id in PENDING:
            return gr.update(), gr.update(), 'המתיני לסיום התמלול הפעיל.'
        rows = [dict(r) for r in job['speech_segments']]
    audio_path = audio_path_for_view(job_id)
    if not audio_path:
        return gr.update(), gr.update(), 'קובץ ההקלטה המקורי אינו זמין בתיקיית Downloads.'
    try:
        assigned, names = diarize_speech(audio_path, rows, mode, int(count))
    except (Exception) as exc:
        return gr.update(), gr.update(), 'ניתוח הדוברים נכשל: ' + str(exc)
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return gr.update(), gr.update(), 'התמלול אינו קיים.'
        job['speaker_segments'], job['speaker_names'] = assigned, names
        job['speaker_mode'], job['speaker_count'] = mode, int(count)
        job['speaker_error'] = ''
        job['text'] = speaker_plain_text(job)
        save_job(job)
        return speaker_reader_html(job), job['text'], f'✅ בוצע ניתוח מחדש. זוהו {len(names)} דוברים. רענני את התמלול כדי לעדכן את הרשימות.'

CSS += r"""
#speaker-editor { border: 1px solid #565d65; border-radius: 12px; padding: 14px; background: #31353a; }
#reader-transcript .speaker-turn { margin: 0 0 20px; line-height: 1.9; }
#reader-transcript .speaker-turn strong { display: block; font-weight: 750; color: #a8e6b6; }
#reader-transcript .speaker-turn div { white-space: pre-wrap; }
"""

def process_job(job_id):
    with LOCK:
        job = JOBS[job_id]
        job["state"] = "מכין תמלול"
        job["percent"] = 0
        job["started_at"] = time.monotonic()
        save_job(job)
        path = job["path"]
        models = [job["model"]]
        if job["compare"]:
            models.append(job["other"])
        context = job["context"]
        clean = job["clean"]
        speaker_enabled = bool(job.get("speaker_enabled", False))
        speaker_mode = job.get("speaker_mode", "אוטומטי")
        speaker_count = int(job.get("speaker_count") or 2)

    if clean:
        with LOCK:
            job["state"] = "מנקה רעשים"
            save_job(job)
        path = clean_audio(path, WORK_DIR / f"{job_id}-clean.wav")

    for model_index, model_name in enumerate(models):
        if STOP_CURRENT.is_set():
            break
        with LOCK:
            job["state"] = f"טוען מודל ({model_name})"
            save_job(job)
        model = get_model(model_name)
        with LOCK:
            job["state"] = f"מתמלל ({model_name})"
            save_job(job)
        options = {
            "language": "he", "beam_size": 2, "temperature": 0,
            "condition_on_previous_text": False, "vad_filter": True,
        }
        if context:
            options["initial_prompt"] = context
        segments, info = model.transcribe(path, **options)
        duration = max(float(info.duration or 1), 1.0)
        with LOCK:
            job["audio_duration_seconds"] = duration
            save_job(job)
        parts = []
        timed_parts = []
        last_saved = time.monotonic()
        for segment in segments:
            if STOP_CURRENT.is_set():
                break
            speech = segment.text.strip()
            if speech:
                parts.append(speech)
                if model_index == 0:
                    timed_parts.append({"start": float(segment.start), "end": float(segment.end), "text": speech})
            fraction = min(max(float(segment.end) / duration, 0.0), 1.0)
            with LOCK:
                job["percent"] = min(99, round(100 * (model_index + fraction) / len(models)))
                if model_index == 0:
                    job["text"] = make_paragraphs(parts)
                else:
                    job["comparison"] = make_paragraphs(parts)
                if time.monotonic() - last_saved >= 5:
                    save_job(job)
                    last_saved = time.monotonic()
        if not parts and not STOP_CURRENT.is_set():
            with LOCK:
                if model_index == 0:
                    job["text"] = "לא זוהה דיבור בהקלטה."
                else:
                    job["comparison"] = "לא זוהה דיבור בהקלטה."
        with LOCK:
            if model_index == 0:
                job['speech_segments'] = timed_parts
            save_job(job)
        del model

    if speaker_enabled and not STOP_CURRENT.is_set():
        with LOCK:
            job['state'] = 'מזהה דוברים'
            save_job(job)
        try:
            speaker_rows, speaker_names = diarize_speech(path, job.get('speech_segments') or [],
                                                          speaker_mode, speaker_count)
            with LOCK:
                job['speaker_segments'] = speaker_rows
                job['speaker_names'] = speaker_names
                job['speaker_error'] = ''
                job['text'] = speaker_plain_text(job)
                save_job(job)
        except Exception as exc:
            # Diarization is optional. Never lose a successful Whisper transcript.
            with LOCK:
                job['speaker_error'] = str(exc)
                save_job(job)

    with LOCK:
        if job.get("started_at") is not None:
            job["elapsed_seconds"] = round(time.monotonic() - job["started_at"], 1)
        job["started_at"] = None
        if STOP_CURRENT.is_set():
            job["state"] = "נעצר"
        else:
            job["state"] = "הושלם"
            job["percent"] = 100
        save_job(job)
        to_upload = dict(job) if job["state"] == "הושלם" else None
    if to_upload:
        with LOCK:
            job["drive_status"] = "מעלה ל־Google Drive" if drive_configured() else "ממתין לחיבור Google Drive"
            save_job(job)
        status = drive_upload(to_upload)
        with LOCK:
            job["drive_status"] = status
            save_job(job)
    gc.collect()


def background_worker():
    global ACTIVE, WORKER_RUNNING
    while True:
        with LOCK:
            if not PENDING:
                ACTIVE = None
                WORKER_RUNNING = False
                return
            job_id = PENDING.popleft()
            ACTIVE = job_id
            STOP_CURRENT.clear()
        try:
            process_job(job_id)
        except Exception as error:
            with LOCK:
                failed = JOBS[job_id]
                if failed.get("started_at") is not None:
                    failed["elapsed_seconds"] = round(time.monotonic() - failed["started_at"], 1)
                failed["started_at"] = None
                failed["state"] = "שגיאה"
                failed["error"] = str(error)
                save_job(failed)
        finally:
            with LOCK:
                ACTIVE = None


def enqueue(files, model, compare, other, clean, context, speaker_enabled=False, speaker_mode="אוטומטי", speaker_count=2):
    global WORKER_RUNNING
    if not files:
        return gr.update(), "לא נבחרו קבצים. בחרי הקלטה ואז לחצי על תמלל."
    if compare and model == other:
        return gr.update(), "להשוואה יש לבחור שני מודלים שונים."
    if not isinstance(files, list):
        files = [files]
    with LOCK:
        if len(PENDING) + len(files) > MAX_PENDING:
            return gr.update(), f"אפשר להמתין עם עד {MAX_PENDING} הקלטות בתור."
        fresh_files = [f for f in files if str(f) not in SESSION_UPLOADS]
    if not fresh_files:
        return gr.update(), "הקבצים האלה כבר נוספו. אפשר לראות אותם בלשונית 'ההקלטות שלי'."
    copied = []
    try:
        for uploaded in fresh_files:
            source = Path(str(uploaded))
            if not source.is_file():
                raise FileNotFoundError(f"הקובץ אינו זמין: {source.name}")
            job_id = uuid.uuid4().hex
            destination = WORK_DIR / f"{job_id}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            # Keep only a size and fingerprint in SQLite, never the audio bytes.
            digest = hashlib.sha256()
            with destination.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            copied.append((job_id, destination, source.name, str(uploaded),
                           destination.stat().st_size, digest.hexdigest()))
    except Exception as error:
        for _, destination, _, _, _, _ in copied:
            destination.unlink(missing_ok=True)
        return gr.update(), "לא ניתן להוסיף את הקבצים: " + str(error)
    with LOCK:
        for job_id, destination, original_name, upload_source, audio_size, audio_hash in copied:
            JOBS[job_id] = {
                "id": job_id, "path": str(destination), "name": original_name,
                "upload_source": upload_source,
                "audio_original_name": original_name,
                "audio_source_size": audio_size, "audio_source_sha256": audio_hash,
                "model": model,
                "compare": bool(compare), "other": other, "clean": bool(clean),
                "context": (context or "").strip(), "state": "ממתין",
                "speaker_enabled": bool(speaker_enabled),
                "speaker_mode": speaker_mode, "speaker_count": int(speaker_count),
                "speech_segments": [], "speaker_segments": [], "speaker_names": {}, "speaker_error": "",
                "started_at": None, "elapsed_seconds": 0,
                "audio_duration_seconds": None, "percent": 0, "text": "", "comparison": "",
                "error": "", "drive_status": "טרם סונכרן",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            SESSION_UPLOADS.add(upload_source)
            save_job(JOBS[job_id])
            PENDING.append(job_id)
        if not WORKER_RUNNING:
            WORKER_RUNNING = True
            threading.Thread(target=background_worker, daemon=True).start()
    return gr.update(), f"✅ נוספו {len(copied)} הקלטות לתור. התמלול נשמר אוטומטית בהיסטוריה המקומית."


def stop_active():
    with LOCK:
        if ACTIVE is None:
            return "אין כרגע הקלטה פעילה לעצירה."
        STOP_CURRENT.set()
        return "התבקשה עצירה. התמלול ייעצר לאחר המקטע הנוכחי והתור ימשיך."


def refresh(selected, last_player_id):
    with LOCK:
        items = list(JOBS.values())
        if not items:
            return (
                gr.update(choices=[], value=None),
                "אין עדיין הקלטות בתור.",
                "הוסיפי הקלטות בלשונית 'תמלול חדש'.",
                None,
                "",
                gr.update(value="", visible=False),
                None,
                progress_card(None, "בחרי הקלטה כדי לראות את ההתקדמות"),
                progress_card(None, "עדיין לא נוספו הקלטות"),
            )

        choices = [
            (f"{item['name']} · {item['state']} · {item['percent']}%", item["id"])
            for item in items
        ]
        ids = {item["id"] for item in items}
        if selected not in ids:
            selected = items[-1]["id"]

        job = JOBS[selected]
        completed = sum(item["state"] == "הושלם" for item in items)
        waiting = sum(item["state"] == "ממתין" for item in items)
        active = sum(item["state"].startswith("מתמלל") or item["state"] == "מנקה רעשים" for item in items)

        summary = (
            f"**{len(items)} הקלטות** · "
            f"{completed} הושלמו · {active} בעבודה · {waiting} ממתינות"
        )
        detail = f"**{job['name']}** · {job['state']} · {job['percent']}%"
        if job["error"]:
            detail += " · שגיאה: " + html.escape(job["error"])
        detail += " · ענן: " + html.escape(job.get("drive_status", "טרם סונכרן"))
        if job.get("elapsed_seconds") and job.get("state") in ("הושלם", "נעצר", "שגיאה"):
            seconds = int(job["elapsed_seconds"])
            detail += f" · ⏱ זמן תמלול: {seconds // 60:02d}:{seconds % 60:02d}"

        active_job = next((item for item in items if item["id"] == ACTIVE), None)
        latest_job = active_job or items[-1]

        return (
            gr.update(choices=choices, value=selected),
            summary,
            detail,
            (job["path"] if job.get("path") and Path(job["path"]).exists() else None)
            if selected != last_player_id else gr.update(),
            job["text"],
            gr.update(value=job["comparison"], visible=job["compare"]),
            selected,
            progress_card(job),
            progress_card(latest_job),
        )




# WHISPER_UI_DESIGN_V1
# These helpers affect only file selection and presentation, not the worker.
def pending_file_size(file_path):
    try:
        size = Path(str(file_path)).stat().st_size
        return f"{size / (1024 * 1024):.2f} MB"
    except OSError:
        return "גודל לא זמין"


def remove_pending_file(target):
    def remove(files):
        return [f for f in (files or []) if str(f) != target]
    return remove


def enqueue_from_ui(files, model, compare, other, clean, context, speaker_enabled, speaker_mode, speaker_count):
    result, message = enqueue(files, model, compare, other, clean, context, speaker_enabled, speaker_mode, speaker_count)
    # Clear the selection only after successfully placing files in the queue.
    if isinstance(message, str) and message.startswith("✅"):
        return gr.update(value=None), message
    return result, message


CSS += r"""
/* New dark, centered UI. Existing upload hover glow is preserved. */
html, body { background: #202020 !important; }
.gradio-container {
    background: #202020 !important;
    color: #e5e5e5 !important;
    min-height: 100vh !important;
    max-width: 1120px !important;
    padding: 12px 18px 50px !important;
}
#app-header { text-align: center; margin: 12px auto 6px; }
#app-header h1 { font-size: 26px !important; font-weight: 650 !important; }
#app-subtitle { text-align: center; opacity: .78; font-size: 13px; margin-bottom: 14px; }
#main-tabs { direction: rtl; }
#main-tabs button.selected, #main-tabs button[aria-selected="true"] {
    border-bottom: 3px solid #58bd6b !important;
    color: #70d281 !important;
}
#transcription-card {
    max-width: 720px !important;
    margin: 16px auto 18px !important;
    padding: 24px !important;
    border: 1px solid #555 !important;
    border-radius: 18px !important;
    background: #303030 !important;
    box-shadow: none !important;
}
#transcription-card .block { border-radius: 10px; }
#transcription-card #audio-upload { background: #393939 !important; }
#transcription-card #audio-upload:hover,
#transcription-card #audio-upload:focus-within {
    border-color: #ff7a00 !important;
}
.pending-file-row {
    background: #3a3a3a !important;
    border: 1px solid #555 !important;
    border-radius: 9px !important;
    padding: 8px 12px !important;
    margin: 4px 0 !important;
    align-items: center !important;
}
.pending-file-text { direction: rtl; text-align: right; padding: 4px; }
.pending-file-name { font-size: 15px; font-weight: 650; overflow-wrap: anywhere; }
.pending-file-size { font-size: 12px; color: #b8b8b8; margin-top: 3px; }
.pending-remove button, button.pending-remove {
    background: #ef5350 !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
}
#transcribe-button { background: #56b96b !important; color: #fff !important; }
#transcribe-button:hover { background: #4ba95e !important; }
#transcribe-button:disabled { opacity: .5 !important; }
#recordings-panel { max-width: 1040px !important; margin: 12px auto !important; }
#active-status { max-width: 720px !important; margin: 0 auto !important; }
@media (max-width: 680px) {
    #transcription-card { padding: 12px !important; }
    .gradio-container { padding: 8px !important; }
}
"""



# WHISPER_LIBRARY_V2
# The original transcription worker and Google Drive synchronization stay intact.
# Delete operations affect the local SQLite row and in-memory history ONLY.

# WHISPER_VIEW_POLISH_V1
# Format stored local wall-clock dates without altering their original values.
def format_history_date(value):
    if not value:
        return ""
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime("%d.%m.%Y בשעה %H:%M")
    except (ValueError, TypeError):
        return str(value)


def library_reader_progress(job):
    # The full orange bar is useful while working, not after completion.
    if job.get("state") != "הושלם":
        return progress_card(job)
    elapsed = int(job.get("elapsed_seconds") or 0)
    duration = f"{elapsed // 60:02d}:{elapsed % 60:02d}" if elapsed else "לא נמדד"
    audio_seconds = job.get("audio_duration_seconds")
    speed = ""
    if elapsed and audio_seconds:
        try:
            speed = f" · מהירות פי {float(audio_seconds) / elapsed:.1f} מזמן אמת"
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    return (
        '<div class="whisper-progress" style="padding:9px 14px;margin:5px 0 12px">'
        '<span style="color:#74ca8c;font-weight:600">✓ התמלול הושלם</span>'
        f'<span style="color:#c4c4cb"> · זמן תמלול: {duration}{speed}</span>'
        '</div>'
    )


# WHISPER_READER_V1
# HTML is always escaped: transcript text is untrusted and must not execute markup.
def reader_text_html(text):
    safe = html.escape(str(text or ""))
    if not safe.strip():
        safe = '<span class="reader-empty">אין טקסט להצגה בתמלול הזה.</span>'
    return f'<article class="reader-text-content" dir="rtl">{safe}</article>'


def reader_plain_text(job_id):
    with LOCK:
        return str(JOBS.get(job_id, {}).get("text", "")) if job_id else ""


def reader_begin_edit(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return gr.update(visible=False), "", "⚠️ בחרי תמלול לעריכה."
        if job_id == ACTIVE or job_id in PENDING:
            return gr.update(visible=False), "", "⚠️ לא ניתן לערוך תמלול פעיל או ממתין."
        return gr.update(visible=True), str(job.get("text", "")), ""


def reader_save_edit(job_id, new_text):
    if not job_id:
        return gr.update(), gr.update(), gr.update(), "⚠️ לא נבחר תמלול."
    new_text = str(new_text or "")
    if not new_text.strip():
        return gr.update(), gr.update(), gr.update(), "⚠️ התמלול ריק. לא בוצעה שמירה."
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return gr.update(), gr.update(), gr.update(), "⚠️ התמלול אינו קיים."
        if job_id == ACTIVE or job_id in PENDING:
            return gr.update(), gr.update(), gr.update(), "⚠️ לא ניתן לשמור בזמן תמלול."
        previous = job.get("text", "")
        job["text"] = new_text
        previous_segments = job.get('speaker_segments')
        job['speaker_segments'] = []  # Free-form edits no longer align with individual segments.
        try:
            save_job(job)
        except Exception as exc:
            job["text"] = previous
            job["speaker_segments"] = previous_segments
            return gr.update(), gr.update(), gr.update(), "❌ השמירה נכשלה: " + str(exc)
    return (gr.update(visible=False), reader_text_html(new_text), new_text,
            "✅ השינויים נשמרו במחשב. העותק הקודם ב־Google Drive לא השתנה.")


CSS += r"""
/* WHISPER_READER_V1: natural page scrolling and readable paragraphs. */
#reader-panel { max-width: 1120px !important; }
#reader-transcript, #reader-comparison { overflow: visible !important; }
#reader-transcript .reader-text-content,
#reader-comparison .reader-text-content {
    direction: rtl; text-align: right; white-space: pre-wrap;
    overflow-wrap: anywhere; word-break: normal;
    line-height: 2.0; font-size: 18px; font-weight: 400;
    color: #f0f1f4; background: #374151;
    border: 1px solid #495565; border-radius: 10px;
    padding: 22px 26px; margin: 4px 0 18px;
    height: auto !important; max-height: none !important; overflow: visible !important;
    user-select: text;
}
#reader-actions { direction: rtl; gap: 10px !important; margin: 8px 0 !important; }
#reader-actions button { min-width: 135px; }
#reader-edit-panel { background: #30363f; border: 1px solid #586575;
    padding: 14px; border-radius: 10px; }
#reader-edit-panel textarea { line-height: 1.8; text-align: right; direction: rtl; }
@media (max-width: 680px) {
    #reader-transcript .reader-text-content,
    #reader-comparison .reader-text-content { padding: 14px; font-size: 16px; }
    #reader-actions button { min-width: 95px; }
}
"""


# WHISPER_LIBRARY_SORT_COMPACT_V1
# Sorting changes presentation order only. IDs, saved records, and jobs are untouched.
def library_sorted_items(items, mode):
    items = list(items or [])
    if mode == "ישן לחדש":
        return sorted(items, key=lambda item: (str(item[3]), str(item[0])))
    if mode == "שם: א׳ עד ת׳":
        return sorted(items, key=lambda item: (str(item[1]).casefold(), str(item[0])))
    if mode == "שם: ת׳ עד א׳":
        return sorted(items, key=lambda item: (str(item[1]).casefold(), str(item[0])), reverse=True)
    # Default is most recent first, including records added during this session.
    return sorted(items, key=lambda item: (str(item[3]), str(item[0])), reverse=True)


CSS += r"""
/* WHISPER_LIBRARY_SORT_COMPACT_V1: tighter library rows only. */
#library-sortbar { max-width: 1050px; margin: 4px auto 0 !important; }
#library-sortbar .form { min-height: 36px !important; }
#library-list { margin-top: 5px !important; }
#library-list .history-row {
    padding: 3px 8px !important;
    margin: 2px 0 !important;
    gap: 6px !important;
    min-height: 48px !important;
}
#library-list button.history-open {
    font-size: 13px !important;
    line-height: 1.4 !important;
    min-height: 38px !important;
    padding: 4px 8px !important;
}
#library-list button.history-delete,
#library-list button.history-rename {
    min-height: 34px !important;
    min-width: 38px !important;
    padding: 4px 8px !important;
}
#library-list .history-action-col { min-width: 42px !important; }
"""


def library_snapshot():
    with LOCK:
        jobs = list(JOBS.values())
        jobs.sort(key=lambda j: (j.get("created_at", ""), j.get("id", "")), reverse=True)
        return [
            (j["id"], j.get("name", "הקלטה"), j.get("state", ""),
             j.get("created_at", ""), int(j.get("percent", 0)))
            for j in jobs
        ]


# WHISPER_LIBRARY_IVRIT_STYLE_V1
HEBREW_MONTHS = (
    "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
    "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
)


def format_library_date_he(value):
    try:
        date = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return format_history_date(value)
    return f"{date.day} ב{HEBREW_MONTHS[date.month - 1]} {date.year} בשעה {date:%H:%M}"


def library_summary():
    with LOCK:
        jobs = list(JOBS.values())
        completed = sum(j.get("state") == "הושלם" for j in jobs)
        in_progress = sum(j.get("id") == ACTIVE for j in jobs)
        return f"**{len(jobs)} הקלטות** · {completed} הושלמו · {in_progress} בעבודה"


def library_live_status():
    with LOCK:
        job = JOBS.get(ACTIVE)
        if job is None and JOBS:
            job = next(reversed(JOBS.values()))
        return progress_card(job, "עדיין לא התחיל תמלול")


# WHISPER_VIEW_FIX_V1
# Populate the reader in the SAME click event as the tab switch.
# This avoids relying on a State.change event that may not fire in time.

# WHISPER_CUSTOM_AUDIO_PLAYER_V1
# Audio is served only from the app's temporary work directory and is NOT
# copied into the persistent SQLite history or Google Drive.

def reader_audio_player(path):
    if not path:
        return ""
    try:
        audio_path = Path(path).resolve(strict=True)
        audio_path.relative_to(WORK_DIR.resolve())
        if not audio_path.is_file():
            return ""
    except (ValueError, OSError):
        return ""
    # This endpoint is permitted by the app's existing allowed_paths setting.
    audio_url = "/gradio_api/file=" + quote(str(audio_path), safe="/")
    audio_url = html.escape(audio_url, quote=True)
    return f"""<section class="whisper-player" dir="ltr" aria-label="נגן ההקלטה">
      <audio class="whisper-player-media" src="{audio_url}" preload="metadata"></audio>
      <div class="whisper-player-buttons">
        <button type="button" class="whisper-player-skip" data-whisper-action="back"
                aria-label="חזרה 10 שניות" title="חזרה 10 שניות">
          <svg viewBox="0 0 48 48" aria-hidden="true"><path d="M10 8v32h5V8zM37 8 16 24l21 16z"/></svg>
        </button>
        <button type="button" class="whisper-player-play" data-whisper-action="toggle"
                aria-label="ניגון" title="ניגון או השהיה">
          <svg class="whisper-icon-play" viewBox="0 0 48 48" aria-hidden="true"><path d="M17 10v28l23-14z"/></svg>
          <svg class="whisper-icon-pause" viewBox="0 0 48 48" aria-hidden="true"><path d="M15 11h7v26h-7zM27 11h7v26h-7z"/></svg>
        </button>
        <button type="button" class="whisper-player-skip" data-whisper-action="forward"
                aria-label="קדימה 10 שניות" title="קדימה 10 שניות">
          <svg viewBox="0 0 48 48" aria-hidden="true"><path d="M33 8h5v32h-5zM11 8l21 16-21 16z"/></svg>
        </button>
      </div>
      <div class="whisper-player-timeline">
        <span class="whisper-player-current">0:00</span>
        <input type="range" min="0" max="1000" value="0" step="1"
               class="whisper-player-seek" aria-label="מעבר למיקום בהקלטה" />
        <span class="whisper-player-duration">0:00</span>
      </div>
      <p class="whisper-player-error" role="status" hidden></p>
    </section>"""


CSS += r"""
/* WHISPER_CUSTOM_AUDIO_PLAYER_V1: large, high-contrast transport controls. */

/* WHISPER_PLAYER_SWAP_RTL_V1: forward on left, back on right. */
#custom-audio-controls .whisper-player-buttons [data-whisper-action="forward"] { order: 1; }
#custom-audio-controls .whisper-player-buttons [data-whisper-action="toggle"] { order: 2; }
#custom-audio-controls .whisper-player-buttons [data-whisper-action="back"] { order: 3; }

#custom-audio-controls { margin: 8px 0 16px; }
#custom-audio-controls .whisper-player {
    border-radius: 16px; padding: 12px 18px 14px;
    background: #28282b; border: 1px solid #4b4b50;
    text-align: center; color: #fff;
}
#custom-audio-controls .whisper-player-media { display: none; }
#custom-audio-controls .whisper-player-buttons {
    display: flex; align-items: center; justify-content: center; gap: clamp(26px, 7vw, 84px);
}
#custom-audio-controls button { cursor: pointer; border: none; transition: transform .15s ease, opacity .15s ease; }
#custom-audio-controls button:hover { transform: scale(1.07); }
#custom-audio-controls button:focus-visible { outline: 3px solid #78d58d; outline-offset: 4px; }
#custom-audio-controls .whisper-player-skip {
    width: 62px; height: 62px; background: transparent; color: #fff; padding: 8px;
}
#custom-audio-controls .whisper-player-skip svg { width: 100%; height: 100%; fill: currentColor; }
#custom-audio-controls .whisper-player-play {
    width: 108px; height: 108px; border-radius: 50%; background: #fff;
    color: #0d0d0f; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 14px rgba(0,0,0,.2);
}
#custom-audio-controls .whisper-player-play svg { width: 51px; height: 51px; fill: currentColor; }
#custom-audio-controls .whisper-icon-pause { display: none; }
#custom-audio-controls .whisper-is-playing .whisper-icon-play { display: none; }
#custom-audio-controls .whisper-is-playing .whisper-icon-pause { display: block; }
#custom-audio-controls .whisper-player-timeline {
    display: flex; gap: 12px; align-items: center; margin-top: 10px; direction: ltr;
    font-size: 13px; color: #dedee2; font-variant-numeric: tabular-nums;
}
#custom-audio-controls .whisper-player-seek { flex: 1; min-width: 30px; accent-color: #79ce8a; cursor: pointer; }
#custom-audio-controls .whisper-player-error { color: #ffd1a8; margin: 6px 0 0; }
/* Keep Gradio's built-in player as a fallback if the custom audio URL fails. */
.whisper-audio-custom-ok #reader-audio { display: none !important; }
@media (max-width: 680px) {
    #custom-audio-controls .whisper-player-play { width: 82px; height: 82px; }
    #custom-audio-controls .whisper-player-play svg { width: 42px; height: 42px; }
    #custom-audio-controls .whisper-player-skip { width: 48px; height: 48px; }
    #custom-audio-controls .whisper-player-buttons { gap: 22px; }
}
"""

# Delegated events survive Gradio rerendering the HTML component on selection.
CUSTOM_AUDIO_JS = r"""() => {
    document.documentElement.classList.add('dark');
    if (window.__whisperCustomAudioInstalled) return;
    window.__whisperCustomAudioInstalled = true;
    const root = document.documentElement;
    const fmt = (sec) => {
        if (!Number.isFinite(sec) || sec < 0) return '0:00';
        const s = Math.floor(sec);
        const h = Math.floor(s / 3600);
        const m = Math.floor(s / 60) % 60;
        return h ? `${h}:${String(m).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`
                 : `${m}:${String(s%60).padStart(2,'0')}`;
    };
    const getPlayer = (el) => el?.closest('.whisper-player');
    const update = (player) => {
        if (!player) return;
        const media = player.querySelector('audio');
        if (!media) return;
        const duration = Number.isFinite(media.duration) ? media.duration : 0;
        player.classList.toggle('whisper-is-playing', !media.paused && !media.ended);
        const toggle = player.querySelector('[data-whisper-action="toggle"]');
        if (toggle) toggle.setAttribute('aria-label', media.paused ? 'ניגון' : 'השהיה');
        const current = player.querySelector('.whisper-player-current');
        const total = player.querySelector('.whisper-player-duration');
        const seek = player.querySelector('.whisper-player-seek');
        if (current) current.textContent = fmt(media.currentTime);
        if (total) total.textContent = fmt(duration);
        if (seek && document.activeElement !== seek) {
            seek.value = duration > 0 ? String(Math.round(media.currentTime / duration * 1000)) : '0';
        }
    };
    document.addEventListener('click', async (event) => {
        const button = event.target.closest?.('[data-whisper-action]');
        if (!button) return;
        const player = getPlayer(button);
        const media = player?.querySelector('audio');
        if (!media) return;
        event.preventDefault();
        const action = button.dataset.whisperAction;
        if (action === 'back') media.currentTime = Math.max(0, media.currentTime - 10);
        if (action === 'forward') media.currentTime = Math.min(
            Number.isFinite(media.duration) ? media.duration : Infinity, media.currentTime + 10);
        if (action === 'toggle') {
            if (media.paused) {
                try { await media.play(); }
                catch (err) {
                    const msg = player.querySelector('.whisper-player-error');
                    if (msg) { msg.textContent = 'לא ניתן לנגן כרגע. אפשר לנסות את הנגן החלופי למטה.'; msg.hidden = false; }
                    root.classList.remove('whisper-audio-custom-ok');
                }
            } else media.pause();
        }
        update(player);
    });
    document.addEventListener('input', (event) => {
        if (!event.target.matches?.('.whisper-player-seek')) return;
        const player = getPlayer(event.target);
        const media = player?.querySelector('audio');
        if (media && Number.isFinite(media.duration) && media.duration > 0) {
            media.currentTime = media.duration * Number(event.target.value) / 1000;
            update(player);
        }
    });
    for (const type of ['loadedmetadata', 'durationchange', 'timeupdate', 'play', 'pause', 'ended']) {
        document.addEventListener(type, (event) => {
            const player = getPlayer(event.target);
            if (!player) return;
            if (type === 'loadedmetadata') {
                root.classList.add('whisper-audio-custom-ok');
                const msg = player.querySelector('.whisper-player-error');
                if (msg) msg.hidden = true;
            }
            update(player);
        }, true);
    }
    document.addEventListener('error', (event) => {
        const player = getPlayer(event.target);
        if (!player) return;
        root.classList.remove('whisper-audio-custom-ok');
        const msg = player.querySelector('.whisper-player-error');
        if (msg) { msg.textContent = 'הנגן המעוצב לא הצליח לטעון את ההקלטה. אפשר להשתמש בנגן החלופי למטה.'; msg.hidden = false; }
    }, true);
    // When opening another transcript, ensure the previous recording stops.
    const observer = new MutationObserver(() => {
        const players = document.querySelectorAll('#custom-audio-controls .whisper-player');
        if (!players.length) root.classList.remove('whisper-audio-custom-ok');
    });
    observer.observe(document.body, {childList: true, subtree: true});
}"""


def library_open(job_id):
    title, audio, custom_player, note, transcript, comparison, progress, meta = library_view(job_id)
    controls = speaker_controls(job_id)
    return (job_id, title, audio, custom_player, note, transcript, comparison,
            progress, meta, gr.update(visible=True), gr.update(selected="view"),
            reader_plain_text(job_id), gr.update(visible=False), "", *controls)


# WHISPER_RECONNECT_DOWNLOADS_AUDIO_V1
# Older transcript records have only their original file name; newer records
# also store size + SHA-256 so a different recording cannot be played by mistake.
# Audio is never saved in SQLite or uploaded to Drive.

def audio_download_directories():
    """Support Downloads, a Hebrew Downloads folder and XDG user-dirs."""
    folders = [Path.home() / "Downloads", Path.home() / "הורדות"]
    user_dirs = Path.home() / ".config" / "user-dirs.dirs"
    try:
        content = user_dirs.read_text(encoding="utf-8")
        match = re.search(r'^XDG_DOWNLOAD_DIR="([^"]+)"', content, re.MULTILINE)
        if match:
            raw = match.group(1).replace("${HOME}", str(Path.home()))
            raw = raw.replace("$HOME", str(Path.home()))
            folders.insert(0, Path(os.path.expanduser(raw)))
    except OSError:
        pass
    unique = []
    for folder in folders:
        try:
            resolved = folder.resolve()
            if resolved.is_dir() and resolved not in unique:
                unique.append(resolved)
        except OSError:
            continue
    return unique


def audio_file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audio_path_for_view(job_id):
    """Return a temporary audio path, restoring it from Downloads if needed."""
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        existing = job.get("path")
        if existing and Path(existing).is_file():
            return existing
        filename = Path(str(job.get("audio_original_name") or job.get("name") or "")).name
        size = job.get("audio_source_size")
        digest = job.get("audio_source_sha256")

    if not filename or Path(filename).suffix.lower() not in {
        ".mp3", ".m4a", ".wav", ".mp4", ".flac", ".ogg"
    }:
        return None

    for folder in audio_download_directories():
        candidate = folder / filename
        try:
            # Do not follow symlinks to files outside the Downloads folder.
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(folder)
            if not resolved.is_file():
                continue
            if size is not None and resolved.stat().st_size != int(size):
                continue
            if digest and audio_file_digest(resolved) != digest:
                continue
            destination = WORK_DIR / f"{job_id}-from-downloads{resolved.suffix.lower()}"
            with LOCK:
                # Another viewer may have already restored the recording.
                job = JOBS.get(job_id)
                if not job:
                    return None
                existing = job.get("path")
                if existing and Path(existing).is_file():
                    return existing
                # Use a session-only copy; never expose all of Downloads in
                # Gradio allowed_paths, and never alter the original recording.
                shutil.copy2(resolved, destination)
                job["path"] = str(destination)
                return str(destination)
        except (OSError, ValueError, TypeError):
            continue
    return None


def library_view(job_id):
    with LOCK:
        job = dict(JOBS.get(job_id, {})) if job_id else {}
    if not job:
        return (
            "### בחרי תמלול מתוך 'הקבצים שלי'",
            gr.update(value=None, visible=False),
            "",
            "קובץ האודיו אינו נשמר לצמיתות באפליקציה.",
            "", gr.update(value="", visible=False),
            progress_card(None, "לא נבחר תמלול"),
            "",
        )
    name = html.escape(job.get("name", "הקלטה"))
    path = audio_path_for_view(job_id)
    available = bool(path and Path(path).is_file())
    audio = gr.update(value=path, visible=True) if available else gr.update(value=None, visible=False)
    audio_note = "" if available else (
        "ℹ️ לא נמצא קובץ שמע תואם בתיקיית ההורדות. "
        "כדי להאזין, השאירי את הקובץ המקורי ב־Downloads באותו שם."
    )
    date = html.escape(format_history_date(job.get("created_at", "")))
    state = html.escape(job.get("state", ""))
    return (
        f"### {name}", audio, reader_audio_player(path) if available else "", audio_note,
        speaker_reader_html(job),
        gr.update(value=reader_text_html(job.get("comparison", "")), visible=bool(job.get("compare"))),
        library_reader_progress(job),
        f"{date} · {state}",
    )


def library_request_delete(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None, gr.update(visible=False), "התמלול כבר אינו קיים."
        if job_id == ACTIVE or job_id in PENDING:
            return None, gr.update(visible=False), "⚠️ אי אפשר למחוק הקלטה בזמן שהיא בתור או בתמלול."
        name = html.escape(job.get("name", "הקלטה"))
    return job_id, gr.update(visible=True), (
        f"**למחוק את '{name}' מהאתר ומהמחשב?** "
        "הפעולה אינה ניתנת לביטול. העותק ב־Google Drive יישאר ללא שינוי."
    )


def library_delete(job_id, viewing_id):
    if not job_id:
        return (None, gr.update(visible=False), library_snapshot(),
                "לא נבחר תמלול למחיקה.", gr.update(), gr.update(), gr.update())
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return (None, gr.update(visible=False), library_snapshot(),
                    "התמלול כבר נמחק.", gr.update(), gr.update(), gr.update())
        if job_id == ACTIVE or job_id in PENDING:
            return (None, gr.update(visible=False), library_snapshot(),
                    "⚠️ אי אפשר למחוק תמלול פעיל או ממתין בתור.", gr.update(), gr.update(), gr.update())
        # The SQL deletion happens before removing the item from the UI.
        try:
            with DB_LOCK, sqlite3.connect(DB_PATH) as db:
                db.execute("PRAGMA secure_delete=ON")
                db.execute("DELETE FROM recordings WHERE id=?", (job_id,))
                db.commit()
        except Exception as exc:
            return (None, gr.update(visible=False), library_snapshot(),
                    "❌ מחיקת התמלול נכשלה: " + str(exc), gr.update(), gr.update(), gr.update())
        JOBS.pop(job_id, None)
        source = job.get("upload_source")
        if source:
            SESSION_UPLOADS.discard(source)
        # Delete only our own temporary copy, never an original upload or cloud file.
        path = job.get("path")
        if path:
            local = Path(path)
            if local.parent == WORK_DIR:
                try:
                    local.unlink(missing_ok=True)
                except OSError:
                    pass
        deleted_view = viewing_id == job_id
        return (
            None, gr.update(visible=False), library_snapshot(),
            "✅ התמלול נמחק מהמחשב ומהאתר. Google Drive לא השתנה.",
            None if deleted_view else gr.update(),
            gr.update(selected="files") if deleted_view else gr.update(),
            gr.update(visible=False) if deleted_view else gr.update(),
        )


def library_request_rename(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None, gr.update(visible=False), ""
        return job_id, gr.update(visible=True), job.get("name", "")


def library_rename(job_id, new_name):
    name = (new_name or "").strip()
    if not job_id or not name:
        return (None, gr.update(visible=False), library_snapshot(),
                "⚠️ יש להזין שם תקין.")
    if len(name) > 180 or any(c in name for c in "\r\n\x00"):
        return (None, gr.update(visible=False), library_snapshot(),
                "⚠️ שם ארוך מדי או מכיל תווים לא תקינים.")
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return (None, gr.update(visible=False), library_snapshot(), "ההקלטה אינה קיימת.")
        old = job.get("name")
        job["name"] = name
        try:
            save_job(job)
        except Exception as exc:
            job["name"] = old
            return (None, gr.update(visible=False), library_snapshot(), "❌ לא ניתן לשנות שם: " + str(exc))
    return (None, gr.update(visible=False), library_snapshot(),
            "✅ שם התמלול עודכן באתר בלבד. העותק ב־Google Drive לא שונה.")


def library_download(job_id):
    with LOCK:
        job = dict(JOBS.get(job_id, {})) if job_id else {}
    if not job or not job.get("text", "").strip():
        return gr.update(value=None, visible=False)
    destination = WORK_DIR / f"transcript-{job_id}.txt"
    destination.write_text(job.get("text", ""), encoding="utf-8")
    return gr.update(value=str(destination), visible=True)


CSS += r"""
/* Transcript library and full-page reader */
#library-list { max-width: 1050px; margin: 12px auto; }
#library-list .history-row {
    background: #363636 !important; border: 1px solid #525252 !important;
    border-radius: 9px !important; padding: 8px 10px !important;
    margin: 5px 0 !important; align-items: center !important;
}
#library-list button.history-open {
    background: transparent !important; color: #e9e9e9 !important;
    border: 0 !important; text-align: right !important;
    white-space: pre-line !important; justify-content: flex-start !important;
    font-size: 15px !important; min-height: 53px !important;
}
#library-list button.history-open:hover { color: #7dd88e !important; }
#library-list button.history-delete {
    background: #6a2729 !important; border: 1px solid #a44a4e !important;
    color: #fff !important; min-width: 46px !important;
}
#library-list button.history-rename {
    background: #404040 !important; border: 1px solid #626262 !important;
    color: #fff !important; min-width: 46px !important;
}
#library-list .history-action-col { min-width: 46px !important; }
#reader-panel { max-width: 1050px; margin: 16px auto; padding: 20px !important;
    border: 1px solid #555 !important; border-radius: 16px !important;
    background: #303030 !important; }
#reader-panel textarea { min-height: 350px; }
#reader-panel #reader-audio { width: 100% !important; }
#delete-confirmation { border: 1px solid #ad5555 !important; border-radius: 12px;
    padding: 14px !important; background: #3a2929 !important; }
#rename-panel { border: 1px solid #5b685d !important; border-radius: 12px;
    padding: 14px !important; }
"""


# WHISPER_LIBRARY_IVRIT_STYLE_V1: compact, right-aligned recording cards.
CSS += r"""
#library-list .history-row {
    direction: rtl !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    min-height: 66px !important;
    margin: 4px 0 !important;
    padding: 7px 10px !important;
    gap: 6px !important;
    background: #393939 !important;
    border: 1px solid #535353 !important;
    border-radius: 9px !important;
}
#library-list .history-open-col {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    position: relative !important;
    justify-content: center !important;
    gap: 0 !important;
}
#library-list .history-open-col .block {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
}
#library-list .history-record-content {
    min-height: 49px;
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 11px;
    text-align: right;
    direction: rtl;
    padding: 2px 3px;
}
#library-list .history-record-details { flex: 1; min-width: 0; }
#library-list .history-record-name {
    font-weight: 700;
    font-size: 15px;
    line-height: 1.45;
    color: #f2f2f2;
    overflow-wrap: anywhere;
}
#library-list .history-record-date {
    color: #a9a9ad;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 2px;
}
#library-list .history-status-dot {
    display: block;
    width: 12px;
    height: 12px;
    min-width: 12px;
    border-radius: 50%;
    background: #9d9da4;
}
#library-list .history-status-dot.history-complete { background: #4bbf66; }
#library-list .history-open-col button.history-open {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    z-index: 2 !important;
    cursor: pointer !important;
}
#library-list .history-open-col:has(button.history-open:hover) .history-record-name {
    color: #84dc98;
}
#library-list .history-open-col:has(button.history-open:focus-visible) {
    outline: 2px solid #80d993;
    outline-offset: 1px;
    border-radius: 5px;
}
#library-list .history-action-col {
    flex: 0 0 40px !important;
    width: 40px !important;
    min-width: 40px !important;
    max-width: 40px !important;
    align-self: center !important;
}
#library-list button.history-delete,
#library-list button.history-rename {
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    max-width: 36px !important;
    padding: 0 !important;
    border-radius: 7px !important;
    font-size: 16px !important;
    line-height: 1 !important;
    margin: auto !important;
}
#library-list button.history-delete {
    background: #642729 !important;
    border: 1px solid #994449 !important;
}
#library-list button.history-rename {
    background: #424242 !important;
    border: 1px solid #666 !important;
}
@media (max-width: 600px) {
    #library-list .history-row { padding: 6px !important; }
    #library-list .history-action-col { flex-basis: 36px !important; min-width: 36px !important; max-width: 36px !important; }
    #library-list button.history-delete, #library-list button.history-rename { width: 32px !important; height: 32px !important; min-width: 32px !important; min-height: 32px !important; }
    #library-list .history-record-name { font-size: 14px; }
}
"""

with gr.Blocks(title="תמלול בעברית", fill_width=True) as app:
    gr.Markdown("# 🎙️ תמלול בעברית", elem_id="app-header")
    gr.Markdown("תמלול מקומי · ההקלטות נשארות במחשב", elem_id="app-subtitle")

    # Client-side state for the selected transcript and the list display.
    viewing_id = gr.State(None)
    view_copy_source = gr.State("")
    history_items = gr.State([])
    delete_candidate = gr.State(None)
    rename_candidate = gr.State(None)

    with gr.Tabs(selected="new", elem_id="main-tabs") as main_tabs:
        with gr.Tab("תמלול חדש", id="new") as new_tab:
            with gr.Column(elem_id="transcription-card"):
                uploads = gr.File(
                    label=None, show_label=False,
                    file_count="multiple", type="filepath",
                    file_types=[".mp3", ".m4a", ".wav", ".mp4", ".flac", ".ogg"],
                    elem_id="audio-upload",
                )

                @gr.render(inputs=[uploads])
                def show_pending_files(files):
                    if not files:
                        return
                    for i, file_path in enumerate(files):
                        full_path = str(file_path)
                        file_name = html.escape(Path(full_path).name)
                        size = pending_file_size(full_path)
                        with gr.Row(elem_classes=["pending-file-row"], key=f"file-row-{i}-{full_path}"):
                            with gr.Column(scale=5, min_width=180):
                                gr.HTML(
                                    f'<div class="pending-file-text">'
                                    f'<div class="pending-file-name">{file_name}</div>'
                                    f'<div class="pending-file-size">{size}</div></div>'
                                )
                            with gr.Column(scale=1, min_width=120):
                                remove_button = gr.Button(
                                    "הסר קובץ", variant="stop", size="sm",
                                    elem_classes=["pending-remove"],
                                )
                                remove_button.click(
                                    fn=remove_pending_file(full_path),
                                    inputs=[uploads], outputs=[uploads], queue=False,
                                )

                with gr.Accordion("⚙️ הגדרות מתקדמות", open=False):
                    with gr.Row():
                        model = gr.Dropdown(["medium", "small"], value="medium", label="מודל ראשי")
                        compare = gr.Checkbox(label="השוואת שני מודלים", value=False)
                        other = gr.Dropdown(["small", "medium"], value="small", label="מודל להשוואה")
                    clean = gr.Checkbox(label="ניקוי רעשים לפני תמלול", value=False)
                    speaker_enabled = gr.Checkbox(label="זיהוי והפרדת דוברים (מקומי)", value=True)
                    with gr.Row():
                        speaker_mode = gr.Dropdown(["אוטומטי", "קבוע"], value="אוטומטי", label="מספר דוברים")
                        speaker_count = gr.Dropdown([str(i) for i in range(1, 9)], value="2",
                                                    label="מספר דוברים קבוע (אם נבחר)")
                    context = gr.Textbox(
                        label="מילות הקשר (לא חובה)",
                        placeholder="שמות פרטיים ומונחים מקצועיים", lines=2,
                    )

                with gr.Row():
                    add = gr.Button("תמלל", variant="primary", elem_id="transcribe-button", interactive=False)
                    stop = gr.Button("■ עצירת ההקלטה הפעילה", variant="secondary")
                uploads.change(
                    lambda files: gr.update(interactive=bool(files)),
                    inputs=[uploads], outputs=[add], queue=False,
                )
                message = gr.Markdown("בחרי הקלטות, בדקי את הרשימה ולחצי על תמלל.")
            with gr.Column(elem_id="active-status"):
                active_progress = gr.HTML(value=progress_card(None, "עדיין לא התחיל תמלול"))

        with gr.Tab("הקבצים שלי", id="files") as recordings_tab:
            with gr.Column(elem_id="recordings-panel"):
                with gr.Row():
                    summary = gr.Markdown("ההיסטוריה המקומית שלך")
                    refresh_button = gr.Button("↻ רענון", size="sm")

                with gr.Accordion("☁️ Google Drive · תמלולים בלבד", open=False):
                    gr.Markdown(
                        "טקסטים בלבד מסונכרנים לתיקייה **התמלולים שלי**. "
                        "מחיקה כאן אינה מוחקת עותקים מ־Google Drive."
                    )
                    with gr.Row():
                        check_drive_button = gr.Button("בדיקת חיבור ל־Drive", size="sm")
                        sync_button = gr.Button("סנכרון תמלולים שלא הועלו", size="sm")
                    drive_message = gr.Markdown("ההיסטוריה המקומית זמינה גם בלי אינטרנט.")

                with gr.Column(visible=False, elem_id="delete-confirmation") as delete_panel:
                    delete_label = gr.Markdown("לא נבחר תמלול למחיקה.")
                    with gr.Row():
                        confirm_delete = gr.Button("כן, למחוק מהמחשב", variant="stop")
                        cancel_delete = gr.Button("ביטול", variant="secondary")

                with gr.Column(visible=False, elem_id="rename-panel") as rename_panel:
                    rename_input = gr.Textbox(label="שם התמלול")
                    with gr.Row():
                        confirm_rename = gr.Button("שמור שם", variant="primary")
                        cancel_rename = gr.Button("ביטול", variant="secondary")

                library_message = gr.Markdown("")
                with gr.Row(elem_id="library-sortbar"):
                    sort_mode = gr.Dropdown(
                        choices=["חדש לישן", "ישן לחדש", "שם: א׳ עד ת׳", "שם: ת׳ עד א׳"],
                        value="חדש לישן", label="מיון תמלולים", interactive=True,
                        elem_id="library-sort",
                    )
                with gr.Column(elem_id="library-list"):
                    @gr.render(inputs=[history_items, sort_mode])
                    def show_library(items, selected_sort):
                        items = library_sorted_items(items, selected_sort)
                        if not items:
                            gr.Markdown("אין תמלולים להצגה. אפשר להתחיל בלשונית 'תמלול חדש'.")
                            return
                        for job_id, name, state, date, percent in items:
                            # Keep the click target separate from the HTML so title/date
                            # can have different styles while retaining Gradio events.
                            status_style = "history-complete" if state == "הושלם" else "history-incomplete"
                            safe_name = html.escape(str(name))
                            safe_date = html.escape(format_library_date_he(date))
                            with gr.Row(elem_classes=["history-row"], key=f"history-{job_id}"):
                                with gr.Column(scale=8, min_width=170, elem_classes=["history-open-col"]):
                                    gr.HTML(
                                        '<div class="history-record-content" dir="rtl">'
                                        f'<span class="history-status-dot {status_style}" aria-hidden="true"></span>'
                                        '<div class="history-record-details">'
                                        f'<div class="history-record-name"><bdi dir="auto">{safe_name}</bdi></div>'
                                        f'<div class="history-record-date">{safe_date}</div>'
                                        '</div></div>'
                                    )
                                    open_button = gr.Button(
                                        f"פתיחת תמלול {name}", size="sm", elem_classes=["history-open"]
                                    )
                                    open_button.click(
                                        fn=lambda target=job_id: library_open(target),
                                        outputs=[viewing_id, view_title, view_audio,
                                                 custom_audio, audio_note, view_text, view_comparison,
                                                 view_progress, view_meta, view_tab, main_tabs,
                                                 view_copy_source, edit_panel, reader_notice,
                                                 speaker_panel, segment_select, segment_speaker,
                                                 speaker_rename_select, speaker_merge_target,
                                                 segment_text, speaker_recount, speaker_notice],
                                        queue=False,
                                    )
                                with gr.Column(scale=0, min_width=40, elem_classes=["history-action-col"]):
                                    rename_button = gr.Button("✎", size="sm", elem_classes=["history-rename"])
                                    rename_button.click(
                                        fn=lambda target=job_id: library_request_rename(target),
                                        outputs=[rename_candidate, rename_panel, rename_input], queue=False,
                                    )
                                with gr.Column(scale=0, min_width=40, elem_classes=["history-action-col"]):
                                    delete_button = gr.Button("🗑", size="sm", elem_classes=["history-delete"])
                                    delete_button.click(
                                        fn=lambda target=job_id: library_request_delete(target),
                                        outputs=[delete_candidate, delete_panel, delete_label], queue=False,
                                    )

        with gr.Tab("צפייה", id="view", visible=False) as view_tab:
            with gr.Column(elem_id="reader-panel"):
                with gr.Row():
                    view_title = gr.Markdown("### בחרי תמלול מתוך 'הקבצים שלי'")
                    back_button = gr.Button("↪ חזרה לרשימה", size="sm")
                view_meta = gr.Markdown("")
                view_audio = gr.Audio(label="האזנה להקלטה · נגן חלופי", interactive=False,
                                      visible=False, elem_id="reader-audio")
                custom_audio = gr.HTML(value="", elem_id="custom-audio-controls")
                audio_note = gr.Markdown("קובץ האודיו אינו נשמר לצמיתות באפליקציה.")
                view_progress = gr.HTML(value=progress_card(None, "לא נבחר תמלול"))
                with gr.Row(elem_id="reader-actions"):
                    copy_button = gr.Button("📋 העתקת טקסט", size="sm")
                    download_button = gr.Button("⬇ הורדת תמלול", size="sm")
                    edit_button = gr.Button("✎ עריכת תמלול", size="sm")
                reader_notice = gr.Markdown("")
                download_file = gr.File(label="קובץ התמלול להורדה", visible=False)
                with gr.Column(visible=False, elem_id="reader-edit-panel") as edit_panel:
                    edit_text = gr.Textbox(label="עריכת הטקסט", lines=16,
                                           max_lines=28, rtl=True)
                    with gr.Row():
                        save_edit_button = gr.Button("💾 שמירת שינויים", variant="primary", size="sm")
                        cancel_edit_button = gr.Button("ביטול", size="sm")
                with gr.Accordion("🗣️ זיהוי דוברים ותיקונים", open=False, visible=False, elem_id="speaker-editor") as speaker_panel:
                    speaker_notice = gr.Markdown("")
                    with gr.Row():
                        speaker_re_mode = gr.Dropdown(["אוטומטי", "קבוע"], value="אוטומטי", label="זיהוי מחדש")
                        speaker_recount = gr.Dropdown([str(i) for i in range(1, 9)], value="2", label="מספר דוברים")
                        speaker_re_button = gr.Button("ניתוח מחדש", size="sm")
                    gr.Markdown("**תיקון מקטע**")
                    segment_select = gr.Dropdown(label="מקטע בתמלול", choices=[])
                    segment_speaker = gr.Dropdown(label="הדובר במקטע", choices=[])
                    segment_text = gr.Textbox(label="טקסט המקטע", lines=3, rtl=True)
                    segment_save = gr.Button("שמירת תיקון המקטע", variant="primary", size="sm")
                    gr.Markdown("**שינוי שם ואיחוד דוברים**")
                    with gr.Row():
                        speaker_rename_select = gr.Dropdown(label="דובר לשינוי שם", choices=[])
                        speaker_new_name = gr.Textbox(label="שם חדש", placeholder="למשל: מטפלת")
                        speaker_rename_button = gr.Button("שינוי שם", size="sm")
                    with gr.Row():
                        speaker_merge_target = gr.Dropdown(label="מיזוג אל הדובר", choices=[])
                        speaker_merge_button = gr.Button("איחוד הדובר שנבחר בשם עם דובר היעד", size="sm")
                view_text = gr.HTML(value=reader_text_html(""), elem_id="reader-transcript")
                view_comparison = gr.HTML(value="", visible=False, elem_id="reader-comparison")

    segment_select.change(speaker_pick_segment, inputs=[viewing_id, segment_select],
                          outputs=[segment_speaker, segment_text], queue=False)
    segment_save.click(speaker_save_segment,
                       inputs=[viewing_id, segment_select, segment_speaker, segment_text],
                       outputs=[view_text, view_copy_source, speaker_notice], queue=False)
    speaker_rename_button.click(speaker_rename,
                                inputs=[viewing_id, speaker_rename_select, speaker_new_name],
                                outputs=[view_text, view_copy_source, speaker_notice], queue=False)
    speaker_merge_button.click(speaker_merge,
                               inputs=[viewing_id, speaker_rename_select, speaker_merge_target],
                               outputs=[view_text, view_copy_source, speaker_notice], queue=False)
    speaker_re_button.click(speaker_reanalyze,
                            inputs=[viewing_id, speaker_re_mode, speaker_recount],
                            outputs=[view_text, view_copy_source, speaker_notice], queue=True)

    # Page actions. A click on a history row opens the viewing tab in-place.
    back_button.click(
        lambda: (gr.update(selected="files"), gr.update(visible=False)),
        outputs=[main_tabs, view_tab], queue=False,
    )
    copy_button.click(
        fn=None, inputs=[], outputs=[reader_notice], queue=False,
        js="""async () => {
            const el = document.querySelector('#reader-transcript');
            const text = (el?.innerText || el?.textContent || '').trim();

            if (!text) return '⚠ לא נמצא תמלול מוצג להעתקה.';

            try {
                await navigator.clipboard.writeText(text);
                return '✅ התמלול הועתק ללוח.';
            } catch (error) {
                return '⚠ הדפדפן חסם את ההעתקה. בדקי את הרשאות הלוח.';
            }
        }""",
    )
    download_button.click(library_download, inputs=[viewing_id],
                          outputs=[download_file], queue=False)
    edit_button.click(reader_begin_edit, inputs=[viewing_id],
                      outputs=[edit_panel, edit_text, reader_notice], queue=False)
    save_edit_button.click(reader_save_edit, inputs=[viewing_id, edit_text],
                           outputs=[edit_panel, view_text, view_copy_source, reader_notice], queue=False)
    cancel_edit_button.click(lambda: (gr.update(visible=False), ""),
                             outputs=[edit_panel, reader_notice], queue=False)
    confirm_delete.click(
        library_delete, inputs=[delete_candidate, viewing_id],
        outputs=[delete_candidate, delete_panel, history_items,
                 library_message, viewing_id, main_tabs, view_tab], queue=False,
    )
    cancel_delete.click(
        lambda: (None, gr.update(visible=False)),
        outputs=[delete_candidate, delete_panel], queue=False,
    )
    confirm_rename.click(
        library_rename, inputs=[rename_candidate, rename_input],
        outputs=[rename_candidate, rename_panel, history_items, library_message],
        queue=False,
    )
    cancel_rename.click(
        lambda: (None, gr.update(visible=False)),
        outputs=[rename_candidate, rename_panel], queue=False,
    )

    added = add.click(
        enqueue_from_ui, inputs=[uploads, model, compare, other, clean, context, speaker_enabled, speaker_mode, speaker_count],
        outputs=[uploads, message], queue=False,
    )
    added.then(library_snapshot, outputs=[history_items], queue=False)
    stop.click(stop_active, outputs=[message], queue=False)
    check_drive_button.click(check_drive, outputs=[drive_message], queue=False)
    sync_button.click(sync_all, outputs=[drive_message], queue=True)
    refresh_button.click(library_snapshot, outputs=[history_items], queue=False)
    recordings_tab.select(
        lambda: (library_snapshot(), gr.update(visible=False)),
        outputs=[history_items, view_tab], queue=False,
    )
    new_tab.select(lambda: gr.update(visible=False), outputs=[view_tab], queue=False)

    app.load(
        lambda: (library_snapshot(), library_summary(), library_live_status()),
        outputs=[history_items, summary, active_progress], queue=False,
    )
    timer = gr.Timer(value=5, active=True)
    timer.tick(
        lambda: (library_snapshot(), library_summary(), library_live_status()),
        outputs=[history_items, summary, active_progress], queue=False,
    )


# WHISPER_PLAYER_DARK_PLAY_SWAP_SIDES_V1
# Force a dark play/pause symbol inside the white circle.
CSS += r"""
#custom-audio-controls button.whisper-player-play {
    background: #ffffff !important;
    color: #28282b !important;
    opacity: 1 !important;
}
#custom-audio-controls button.whisper-player-play svg,
#custom-audio-controls button.whisper-player-play svg path {
    fill: #28282b !important;
    color: #28282b !important;
    stroke: none !important;
    opacity: 1 !important;
    filter: none !important;
}
/* Left: back. Center: play. Right: forward. */
#custom-audio-controls .whisper-player-buttons [data-whisper-action="back"] { order: 1 !important; }
#custom-audio-controls .whisper-player-buttons [data-whisper-action="toggle"] { order: 2 !important; }
#custom-audio-controls .whisper-player-buttons [data-whisper-action="forward"] { order: 3 !important; }
"""

app.queue()
app.launch(
    server_name="127.0.0.1",
    server_port=7860,
    css=CSS,
    theme=gr.themes.Base(primary_hue="green", neutral_hue="gray"),
    js=CUSTOM_AUDIO_JS,
    allowed_paths=[str(WORK_DIR)],
)
