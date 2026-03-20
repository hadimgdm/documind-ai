import os
import re
import json
import hashlib
import time
import random
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from io import BytesIO

import streamlit as st
from openai import OpenAI
import numpy as np
import faiss
from pypdf import PdfReader
import pytesseract
from PIL import Image
import pandas as pd
import bcrypt

# Page configuration
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# Load API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY was not found in the system environment variables.")
    st.info("Please set the key in your Environment Variables and restart VS Code.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Base directory for storing data
BASE_DIR = Path("data")
BASE_DIR.mkdir(exist_ok=True)

# Security storage files
USERS_FILE = BASE_DIR / "users.json"
LOG_FILE = BASE_DIR / "logs.json"
OTP_FILE = BASE_DIR / "otp_store.json"

# Models
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# Main settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
RETRIEVE_CANDIDATES = 12
FINAL_TOP_K = 5

# Security settings
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
MAX_REQUESTS_PER_DAY = 100
ALLOWED_FILE_TYPES = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".xls"}
OTP_EXPIRY_SECONDS = 10 * 60
OTP_RESEND_COOLDOWN_SECONDS = 60

# Set Tesseract path manually if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auth session
if "user" not in st.session_state:
    st.session_state.user = None

if "role" not in st.session_state:
    st.session_state.role = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"

if "otp_sent_email" not in st.session_state:
    st.session_state.otp_sent_email = ""

if "otp_verified_email" not in st.session_state:
    st.session_state.otp_verified_email = ""

if "otp_debug_code" not in st.session_state:
    st.session_state.otp_debug_code = ""

# Generate deterministic workspace ID from email
def generate_workspace_id(email: str) -> str:
    email = email.strip().lower()
    local_part = email.split("@")[0] if "@" in email else email
    local_part = re.sub(r"[^a-zA-Z0-9]+", "-", local_part).strip("-").lower() or "user"
    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()[:10]
    return f"ws-{local_part[:24]}-{digest}"

# Clean client ID
def safe_client_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip())
    return cleaned[:80] or "default_client"

# Create client folder
def client_dir(client_id: str) -> Path:
    path = BASE_DIR / client_id
    path.mkdir(parents=True, exist_ok=True)
    return path

# Create text hash for deduplication and cache
def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# Load JSON
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# Save JSON
def save_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# Normalize text
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =========================
# Security: auth / access
# =========================

# Password policy
def validate_password_policy(password: str) -> list[str]:
    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter (A-Z).")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter (a-z).")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one number (0-9).")
    if not re.search(r"[!@#$%^&*]", password):
        errors.append("Password must contain at least one symbol (!@#$%^&*).")

    return errors

# OTP storage
def load_otp_store() -> dict:
    data = load_json(OTP_FILE, {})
    return data if isinstance(data, dict) else {}

def save_otp_store(store: dict) -> None:
    save_json(OTP_FILE, store)

def cleanup_expired_otp(store: dict | None = None) -> dict:
    store = store if store is not None else load_otp_store()
    now = time.time()
    cleaned = {
        email: item for email, item in store.items()
        if now - item.get("created_at", 0) <= OTP_EXPIRY_SECONDS
    }
    if cleaned != store:
        save_otp_store(cleaned)
    return cleaned

def send_signup_otp(email: str) -> tuple[bool, str, str]:
    email = email.strip().lower()
    users = load_users()

    if not email:
        return False, "Email is required.", ""
    if "@" not in email:
        return False, "Please enter a valid email address.", ""
    if email in users:
        return False, "This email is already registered.", ""

    store = cleanup_expired_otp()
    now = time.time()
    existing = store.get(email)

    if existing and now - existing.get("last_sent_at", 0) < OTP_RESEND_COOLDOWN_SECONDS:
        wait_seconds = int(OTP_RESEND_COOLDOWN_SECONDS - (now - existing.get("last_sent_at", 0)))
        return False, f"Please wait {wait_seconds} seconds before requesting a new code.", ""

    otp_code = f"{random.randint(0, 999999):06d}"
    store[email] = {
        "code": otp_code,
        "created_at": now,
        "last_sent_at": now,
        "attempts": 0
    }
    save_otp_store(store)

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "noreply@example.com")

    if smtp_host and smtp_user and smtp_password:
        try:
            msg = EmailMessage()
            msg["Subject"] = "Your DocuMind AI verification code"
            msg["From"] = smtp_from
            msg["To"] = email
            msg.set_content(
                "Your DocuMind AI verification code is: "
                f"{otp_code}\n\n"
                "This code expires in 10 minutes."
            )

            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            log_event(email, "otp_sent", "smtp")
            return True, "Verification code sent to your email.", ""
        except Exception as e:
            log_event(email, "otp_send_fallback", str(e)[:200])

    log_event(email, "otp_sent", "fallback_test_mode")
    return True, "SMTP is not configured. Test-mode OTP generated.", otp_code

def verify_signup_otp(email: str, otp_code: str) -> tuple[bool, str]:
    email = email.strip().lower()
    otp_code = otp_code.strip()
    store = cleanup_expired_otp()
    item = store.get(email)

    if not item:
        return False, "No active OTP found for this email. Please request a new code."

    if item.get("attempts", 0) >= 5:
        store.pop(email, None)
        save_otp_store(store)
        return False, "Too many failed attempts. Please request a new code."

    if time.time() - item.get("created_at", 0) > OTP_EXPIRY_SECONDS:
        store.pop(email, None)
        save_otp_store(store)
        return False, "This OTP has expired. Please request a new code."

    if item.get("code") != otp_code:
        item["attempts"] = item.get("attempts", 0) + 1
        store[email] = item
        save_otp_store(store)
        return False, "Invalid verification code."

    store.pop(email, None)
    save_otp_store(store)
    log_event(email, "otp_verified", "success")
    return True, "Email verified successfully."

# Load users
def load_users() -> dict:
    return load_json(USERS_FILE, {})

# Save users
def save_users(users: dict) -> None:
    save_json(USERS_FILE, users)

# Load logs
def load_logs() -> dict:
    data = load_json(LOG_FILE, {})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("events", [])
    data.setdefault("rate_limits", {})
    return data

# Save logs
def save_logs(logs: dict) -> None:
    save_json(LOG_FILE, logs)

# Log security / usage events
def log_event(user: str, action: str, details: str = "") -> None:
    logs = load_logs()
    logs["events"].append({
        "user": user,
        "action": action,
        "details": details,
        "time": time.time()
    })
    save_logs(logs)

# Register user
def register_user(email: str, password: str, confirm_password: str, email_verified: bool = False) -> tuple[bool, str]:
    email = email.strip().lower()
    users = load_users()

    if not email:
        return False, "Email is required."
    if "@" not in email:
        return False, "Please enter a valid email address."
    if email in users:
        return False, "This email is already registered."

    password_errors = validate_password_policy(password)
    if password_errors:
        return False, " ".join(password_errors)

    if password != confirm_password:
        return False, "Passwords do not match."

    if not email_verified:
        return False, "Please verify your email before completing signup."

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    workspace_id = generate_workspace_id(email)

    users[email] = {
        "password": hashed,
        "role": "owner",
        "allowed_clients": [workspace_id],
        "workspace_id": workspace_id,
        "created_at": time.time(),
        "is_active": True,
        "email_verified": True
    }

    save_users(users)
    log_event(email, "signup", f"account_created | workspace={workspace_id}")
    return True, "Account created successfully. Please log in."

# Login user
def login_user(email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()
    users = load_users()

    if email not in users:
        return False, "Account not found."

    user = users[email]

    if not user.get("is_active", True):
        return False, "This account is disabled."

    hashed = user["password"].encode("utf-8")

    if bcrypt.checkpw(password.encode("utf-8"), hashed):
        st.session_state.user = email
        st.session_state.role = user.get("role", "user")
        st.session_state.otp_verified_email = ""
        st.session_state.otp_sent_email = ""
        st.session_state.otp_debug_code = ""
        log_event(email, "login", "success")
        return True, "Logged in successfully."

    log_event(email, "login", "failed")
    return False, "Invalid email or password."

# Logout user
def logout_user() -> None:
    if st.session_state.user:
        log_event(st.session_state.user, "logout", "manual")
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.messages = []

# Workspace authorization
def authorize_client_access(email: str, client_id: str) -> tuple[bool, str]:
    users = load_users()
    user = users.get(email)

    if not user:
        return False, "User not found."

    role = user.get("role", "user")
    workspace_id = user.get("workspace_id") or generate_workspace_id(email)
    allowed_clients = user.get("allowed_clients", [])

    if role == "superadmin":
        return True, "Access granted."

    if client_id == workspace_id:
        if workspace_id not in allowed_clients:
            user["allowed_clients"] = list(dict.fromkeys([*allowed_clients, workspace_id]))
            user["workspace_id"] = workspace_id
            users[email] = user
            save_users(users)
        return True, "Access granted."

    if client_id in allowed_clients:
        return True, "Access granted."

    if not allowed_clients:
        user["allowed_clients"] = [workspace_id]
        user["workspace_id"] = workspace_id
        users[email] = user
        save_users(users)
        log_event(email, "client_bound", workspace_id)
        return True, "First workspace assigned."

    return False, "You are not allowed to access this workspace."

# File validation
def validate_uploaded_file(file) -> tuple[bool, str]:
    name = file.name
    suffix = Path(name).suffix.lower()
    size = len(file.getvalue())

    if suffix not in ALLOWED_FILE_TYPES:
        return False, f"Unsupported file type: {suffix}"

    if size > MAX_FILE_SIZE_BYTES:
        return False, f"File too large: {name}. Max allowed size is 5 MB."

    return True, "ok"

# Rate limiting
def check_rate_limit(user: str) -> bool:
    logs = load_logs()
    now = time.time()

    user_entries = logs["rate_limits"].get(user, [])
    user_entries = [t for t in user_entries if now - t < 86400]

    if len(user_entries) >= MAX_REQUESTS_PER_DAY:
        logs["rate_limits"][user] = user_entries
        save_logs(logs)
        return False

    user_entries.append(now)
    logs["rate_limits"][user] = user_entries
    save_logs(logs)
    return True

# Management permission
def can_manage_workspace(role: str) -> bool:
    return role in {"owner", "admin", "superadmin"}

# Read TXT file
def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

# Convert Excel to AI-readable text
def extract_text_from_excel(file_bytes: bytes, file_name: str) -> list[dict]:
    segments = []

    try:
        excel_data = pd.read_excel(BytesIO(file_bytes), sheet_name=None)

        for sheet_name, df in excel_data.items():
            if df is None or df.empty:
                continue

            df = df.copy()
            df.columns = [str(c) for c in df.columns]
            df = df.fillna("")

            lines = []
            lines.append(f"Sheet: {sheet_name}")
            lines.append(f"Rows: {len(df)}")
            lines.append(f"Columns: {', '.join(df.columns)}")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                lines.append("Numeric summary:")
                for col in numeric_cols[:15]:
                    try:
                        lines.append(
                            f"- {col}: sum={df[col].sum()}, mean={round(df[col].mean(), 2)}, min={df[col].min()}, max={df[col].max()}"
                        )
                    except Exception:
                        pass

            lines.append("Data:")
            for _, row in df.iterrows():
                row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                lines.append(row_text)

            text = normalize_text("\n".join(lines))

            if text:
                segments.append({
                    "text": text,
                    "source": file_name,
                    "location": f"sheet {sheet_name}"
                })

    except Exception as e:
        segments.append({
            "text": f"Excel parse error: {str(e)}",
            "source": file_name,
            "location": "excel parsing"
        })

    return segments

# Extract text from file
def extract_text(file) -> list[dict]:
    name = file.name
    suffix = Path(name).suffix.lower()
    raw = file.getvalue()

    if len(raw) > MAX_FILE_SIZE_BYTES:
        return [{
            "text": "File too large.",
            "source": name,
            "location": "validation_error"
        }]

    segments = []

    # TXT
    if suffix == ".txt":
        text = normalize_text(extract_text_from_txt(raw))
        if text:
            segments.append({
                "text": text,
                "source": name,
                "location": "full file"
            })
        return segments

    # PDF
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = normalize_text(page_text)

            if page_text:
                segments.append({
                    "text": page_text,
                    "source": name,
                    "location": f"page {page_num}"
                })
        return segments

    # Image
    if suffix in {".png", ".jpg", ".jpeg"}:
        img = Image.open(BytesIO(raw))
        text = pytesseract.image_to_string(img)
        text = normalize_text(text)

        if text:
            segments.append({
                "text": text,
                "source": name,
                "location": "image OCR"
            })
        return segments

    # Excel
    if suffix in {".xlsx", ".xls"}:
        return extract_text_from_excel(raw, name)

    return segments

# Split text into chunks
def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = 0

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks

# Build chunk records
def build_chunk_records(segments: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    records = []
    seen_hashes = set()

    for seg in segments:
        seg_chunks = chunk_text(seg["text"], chunk_size=chunk_size, overlap=overlap)

        for idx, ch in enumerate(seg_chunks, start=1):
            h = text_hash(ch)

            if h in seen_hashes:
                continue

            seen_hashes.add(h)

            records.append({
                "text": ch,
                "source": seg["source"],
                "location": seg["location"],
                "chunk_id": idx,
                "text_hash": h
            })

    return records

# Load embedding cache
def load_embedding_cache(cache_path: Path) -> dict:
    return load_json(cache_path, {})

# Save embedding cache
def save_embedding_cache(cache_path: Path, cache_obj: dict) -> None:
    save_json(cache_path, cache_obj)

# Create embeddings for chunks
def create_embeddings(chunk_records: list[dict], cache_path: Path, progress_placeholder=None) -> np.ndarray:
    cache = load_embedding_cache(cache_path)
    vectors = []
    total = len(chunk_records)

    for n, rec in enumerate(chunk_records, start=1):
        key = rec["text_hash"]

        if key in cache:
            emb = cache[key]
        else:
            r = client.embeddings.create(
                model=EMBED_MODEL,
                input=rec["text"]
            )
            emb = r.data[0].embedding
            cache[key] = emb

        vectors.append(emb)

        if progress_placeholder is not None:
            progress_placeholder.progress(
                n / max(total, 1),
                text=f"Building embeddings... {n}/{total}"
            )

    save_embedding_cache(cache_path, cache)
    return np.array(vectors, dtype="float32")

# Build FAISS index
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    if len(vectors) == 0:
        raise ValueError("No vectors available to build the index.")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# Simple keyword scoring
def keyword_score(query: str, text: str) -> int:
    q_words = [w for w in re.split(r"\W+", query.lower()) if w]

    if not q_words:
        return 0

    text_lower = text.lower()
    score = 0

    for w in q_words:
        if w in text_lower:
            score += 1

    return score

# Retrieve relevant chunks
def retrieve_chunks(index, chunk_records: list[dict], question: str, top_candidates: int = RETRIEVE_CANDIDATES, final_top_k: int = FINAL_TOP_K) -> list[dict]:
    r = client.embeddings.create(
        model=EMBED_MODEL,
        input=question
    )
    q_vec = np.array([r.data[0].embedding], dtype="float32")

    search_k = min(top_candidates, len(chunk_records))
    distances, indices = index.search(q_vec, search_k)

    candidates = []

    for rank, idx in enumerate(indices[0]):
        rec = chunk_records[idx].copy()

        dist = float(distances[0][rank])
        vector_score = 1.0 / (1.0 + dist)
        lexical = keyword_score(question, rec["text"])
        final_score = vector_score * 10 + lexical

        rec["distance"] = dist
        rec["vector_score"] = vector_score
        rec["keyword_score"] = lexical
        rec["final_score"] = final_score

        candidates.append(rec)

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:min(final_top_k, len(candidates))]

# Build context
def build_context(selected_chunks: list[dict]) -> str:
    context_blocks = []

    for i, ch in enumerate(selected_chunks, start=1):
        context_blocks.append(
            f"[Context {i}]\n"
            f"Source: {ch['source']} | {ch['location']} | chunk {ch['chunk_id']}\n"
            f"{ch['text']}"
        )

    return "\n\n".join(context_blocks)

# Answer the question
def answer_with_context(question: str, selected_chunks: list[dict]) -> str:
    context = build_context(selected_chunks)

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only using the provided context. "
                    "If the answer is not clearly present in the context, say exactly: I don't know. "
                    "Do not add anything from yourself. "
                    "Keep the answer precise and concise."
                )
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}"
            }
        ]
    )

    return r.choices[0].message.content

# Generate report
def generate_report_from_chunks(chunk_records: list[dict]) -> str:
    limited_chunks = chunk_records[:20]
    context = "\n\n".join([x["text"] for x in limited_chunks])

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a professional management report based on the provided text. "
                    "The report must include these sections:\n"
                    "1. Overall summary\n"
                    "2. Key points\n"
                    "3. Important numbers and dates\n"
                    "4. Main obligations or requirements\n"
                    "5. Final conclusion"
                )
            },
            {
                "role": "user",
                "content": context
            }
        ]
    )

    return r.choices[0].message.content

# Generate suggestions
def generate_suggestions_from_chunks(chunk_records: list[dict]) -> str:
    limited_chunks = chunk_records[:20]
    context = "\n\n".join([x["text"] for x in limited_chunks])

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Provide improvement suggestions based on the given text. "
                    "Cover these areas:\n"
                    "1. Ambiguous parts\n"
                    "2. Possible missing information\n"
                    "3. Suggested improvements\n"
                    "4. Important decision-making notes"
                )
            },
            {
                "role": "user",
                "content": context
            }
        ]
    )

    return r.choices[0].message.content

# Extract risk highlights
def generate_risk_highlights(chunk_records: list[dict]) -> str:
    limited_chunks = chunk_records[:20]
    context = "\n\n".join([x["text"] for x in limited_chunks])

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract risks and warning points from the given text. "
                    "If there is no clear risk, say so. "
                    "Return the output as a short, management-style bullet list."
                )
            },
            {
                "role": "user",
                "content": context
            }
        ]
    )

    return r.choices[0].message.content

# Generate FAQ
def generate_faq_from_chunks(chunk_records: list[dict]) -> str:
    limited_chunks = chunk_records[:20]
    context = "\n\n".join([x["text"] for x in limited_chunks])

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate 5 to 10 possible frequently asked questions with short answers based on the given text. "
                    "Use only the information inside the text."
                )
            },
            {
                "role": "user",
                "content": context
            }
        ]
    )

    return r.choices[0].message.content

# Save client artifacts
def save_client_artifacts(cdir: Path, index, chunk_records: list[dict]) -> None:
    faiss.write_index(index, str(cdir / "index.faiss"))
    save_json(cdir / "chunks.json", chunk_records)

# Load client artifacts
def load_client_artifacts(cdir: Path):
    index_path = cdir / "index.faiss"
    chunks_path = cdir / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        return None, []

    index = faiss.read_index(str(index_path))
    chunk_records = load_json(chunks_path, [])
    return index, chunk_records

# Clear client data
def reset_client_data(cdir: Path) -> None:
    for name in ["index.faiss", "chunks.json"]:
        p = cdir / name
        if p.exists():
            p.unlink()

# Simple styling
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero-box {
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    background: rgba(99, 102, 241, 0.10);
    border: 1px solid rgba(99, 102, 241, 0.25);
    margin-bottom: 1.2rem;
}
.small-muted {
    opacity: 0.82;
    font-size: 0.95rem;
}
.feature-card {
    padding: 0.9rem 1rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0.7rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero-box">
    <h1 style="margin-bottom:0.35rem;">🧠 DocuMind AI</h1>
    <div class="small-muted">
        Turn your company documents into a smart AI assistant.<br>
        Ask questions Get accurate answers See exact sources
    </div>
</div>
""", unsafe_allow_html=True)

st.info("Step 1: Sign up or log in → Step 2: Your workspace is created automatically from your email → Step 3: Upload files → Step 4: Build Index → Step 5: Ask questions or generate reports")

# Authentication
st.subheader("🔐 Authentication")

auth_toggle_col1, auth_toggle_col2 = st.columns(2)
with auth_toggle_col1:
    if st.button("Login", use_container_width=True, type="primary" if st.session_state.auth_mode == "Login" else "secondary"):
        st.session_state.auth_mode = "Login"

with auth_toggle_col2:
    if st.button("Sign Up", use_container_width=True, type="primary" if st.session_state.auth_mode == "Signup" else "secondary"):
        st.session_state.auth_mode = "Signup"

st.markdown("---")

if st.session_state.auth_mode == "Login":
    login_left, login_right = st.columns([1.2, 1])

    with login_left:
        st.markdown("### Welcome back")
        auth_email = st.text_input("Email", key="login_email")
        auth_password = st.text_input("Password", type="password", key="login_password")

    with login_right:
        st.markdown("### Sign in")
        st.caption("Use your verified account to access your private workspace.")
        st.write("")
        if st.button("Login", use_container_width=True, key="login_submit"):
            ok, msg = login_user(auth_email, auth_password)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

else:
    signup_left, signup_right = st.columns([1.35, 1])

    with signup_left:
        st.markdown("### Create your account")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        otp_code = st.text_input("Email Verification Code (OTP)", key="signup_otp", max_chars=6)

        password_errors = validate_password_policy(signup_password) if signup_password else []
        with st.expander("Password requirements", expanded=True):
            st.markdown("- Minimum 8 characters")
            st.markdown("- At least one uppercase letter (A-Z)")
            st.markdown("- At least one lowercase letter (a-z)")
            st.markdown("- At least one number (0-9)")
            st.markdown("- At least one symbol (!@#$%^&*)")

            if signup_password:
                if password_errors:
                    for err in password_errors:
                        st.error(err)
                else:
                    st.success("Password policy passed.")

    with signup_right:
        st.markdown("### Verify email")
        predicted_workspace = generate_workspace_id(signup_email) if signup_email else "ws-generated-after-email"
        st.caption("Client ID has been removed from the UI.")
        st.info(f"Your workspace will be created automatically: `{predicted_workspace}`")

        send_disabled = not signup_email
        if st.button("Send Verification Code", use_container_width=True, disabled=send_disabled, key="send_otp_btn"):
            ok, msg, debug_code = send_signup_otp(signup_email)
            if ok:
                st.session_state.otp_sent_email = signup_email.strip().lower()
                st.session_state.otp_debug_code = debug_code
                st.success(msg)
                if debug_code:
                    st.warning(f"Test mode OTP: {debug_code}")
            else:
                st.error(msg)

        email_for_verification = signup_email.strip().lower()
        verify_disabled = not email_for_verification or not otp_code

        if st.button("Verify Email", use_container_width=True, disabled=verify_disabled, key="verify_otp_btn"):
            ok, msg = verify_signup_otp(email_for_verification, otp_code)
            if ok:
                st.session_state.otp_verified_email = email_for_verification
                st.success(msg)
            else:
                st.error(msg)

        is_verified_for_signup = st.session_state.otp_verified_email == signup_email.strip().lower() and bool(signup_email.strip())
        if is_verified_for_signup:
            st.success("Email is verified and ready for signup.")
        elif signup_email:
            st.caption("Complete email verification before creating the account.")

        if st.button("Create Account", use_container_width=True, key="signup_submit"):
            ok, msg = register_user(
                signup_email,
                signup_password,
                signup_confirm,
                email_verified=is_verified_for_signup
            )
            if ok:
                st.success(msg)
                st.session_state.auth_mode = "Login"
                st.session_state.otp_sent_email = ""
                st.session_state.otp_verified_email = ""
                st.session_state.otp_debug_code = ""
            else:
                st.error(msg)

if not st.session_state.user:
    st.stop()

login_info_col1, login_info_col2 = st.columns([3, 1])

with login_info_col1:
    st.success(f"Logged in as: {st.session_state.user} | Role: {st.session_state.role}")

with login_info_col2:
    if st.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()

col_x, col_y = st.columns(2)

with col_x:
    st.markdown("""
<div class="feature-card">
<b>🚀 What this product does</b><br>
- Reads TXT, PDF, image, and Excel files<br>
- Answers questions based on your documents<br>
- Shows the exact source of each answer<br>
- Generates management reports<br>
- Provides suggestions and key insights
</div>
""", unsafe_allow_html=True)

with col_y:
    st.markdown("""
<div class="feature-card">
<b>🔐 Why it is reliable</b><br>
- Answers only from your data<br>
- Says "I don't know" when the answer is not in the text<br>
- Keeps each client's data isolated<br>
- Provides source-backed responses
</div>
""", unsafe_allow_html=True)

# Top controls
left, right = st.columns([2, 1])

workspace_id = generate_workspace_id(st.session_state.user)

with left:
    st.markdown("### Workspace")
    st.success(f"Automatic workspace: `{workspace_id}`")
    st.caption("Workspace is generated automatically from the logged-in email. No Client ID input is required.")

with right:
    chunk_size = st.number_input("Chunk Size", min_value=200, max_value=1200, value=500, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=300, value=80, step=10)

client_id = workspace_id

has_access, access_msg = authorize_client_access(st.session_state.user, client_id)
if not has_access:
    st.error(access_msg)
    st.stop()

cdir = client_dir(client_id)
cache_path = cdir / "embedding_cache.json"

# Sidebar
with st.sidebar:
    st.header("Project Status")
    st.write(f"**Product:** `DocuMind AI`")
    st.write(f"**User:** `{st.session_state.user}`")
    st.write(f"**Role:** `{st.session_state.role}`")
    st.write(f"**Workspace ID:** `{client_id}`")
    st.write(f"**Storage Path:** `{cdir}`")

    index, chunk_records = load_client_artifacts(cdir)
    st.write(f"**Indexed Chunks:** `{len(chunk_records)}`")

    st.markdown("---")
    st.markdown("### Supported File Types")
    st.markdown("- TXT")
    st.markdown("- PDF")
    st.markdown("- PNG / JPG / JPEG")
    st.markdown("- XLSX / XLS")

    st.markdown("---")
    st.markdown("### Core Capabilities")
    st.markdown("- OCR")
    st.markdown("- Chunking + Overlap")
    st.markdown("- Deduplication")
    st.markdown("- Embedding Cache")
    st.markdown("- Vector Search")
    st.markdown("- Hybrid Reranking")
    st.markdown("- Source Attribution")
    st.markdown("- Multi-client Isolation")
    st.markdown("- Report")
    st.markdown("- Suggestions")
    st.markdown("- Risk Highlights")
    st.markdown("- FAQ Generator")

# File upload
st.subheader("1) Upload Documents")

uploaded_files = st.file_uploader(
    "Upload TXT, PDF, image, or Excel files",
    type=["txt", "pdf", "png", "jpg", "jpeg", "xlsx", "xls"],
    accept_multiple_files=True
)

col_a, col_b = st.columns(2)

build_clicked = col_a.button("Build / Rebuild Index", use_container_width=True)

clear_clicked = False
if can_manage_workspace(st.session_state.role):
    clear_clicked = col_b.button("Clear Client Data", use_container_width=True)
else:
    col_b.button("Clear Client Data (admin only)", use_container_width=True, disabled=True)

if clear_clicked:
    reset_client_data(cdir)
    st.success(f"Stored data for workspace {client_id} was cleared.")
    st.session_state.messages = []
    log_event(st.session_state.user, "clear_client_data", client_id)

if build_clicked:
    if not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        extraction_progress = st.progress(0, text="Reading files...")
        all_segments = []
        total_files = len(uploaded_files)

        for n, file in enumerate(uploaded_files, start=1):
            valid, validation_message = validate_uploaded_file(file)
            if not valid:
                st.error(validation_message)
                log_event(st.session_state.user, "file_validation_failed", f"{file.name} | {validation_message}")
                continue

            segments = extract_text(file)
            all_segments.extend(segments)
            extraction_progress.progress(
                n / total_files,
                text=f"Reading files... {n}/{total_files}"
            )

        if not all_segments:
            st.error("No readable text could be extracted from the uploaded files.")
        else:
            chunk_records = build_chunk_records(
                all_segments,
                chunk_size=int(chunk_size),
                overlap=int(chunk_overlap)
            )

            if not chunk_records:
                st.error("No chunks were created.")
            else:
                embed_progress = st.progress(0, text="Building embeddings...")
                vectors = create_embeddings(
                    chunk_records,
                    cache_path,
                    progress_placeholder=embed_progress
                )

                try:
                    index = build_faiss_index(vectors)
                    save_client_artifacts(cdir, index, chunk_records)
                    log_event(st.session_state.user, "build_index", client_id)

                    st.success(
                        f"Index was successfully built for workspace '{client_id}'. "
                        f"Number of chunks: {len(chunk_records)}"
                    )
                except Exception as e:
                    st.error(f"Failed to build index: {e}")
                    log_event(st.session_state.user, "build_index_failed", str(e))

# Load saved data
index, chunk_records = load_client_artifacts(cdir)

# Chat section
st.subheader("2) Chat with Documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about this client's documents")

if prompt:
    if not check_rate_limit(st.session_state.user):
        st.error(f"Rate limit exceeded. Max {MAX_REQUESTS_PER_DAY} requests per day.")
        log_event(st.session_state.user, "rate_limit_blocked", client_id)
        st.stop()

    if index is None or not chunk_records:
        st.error("No index exists for this client yet. Upload files and build the index first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            log_event(st.session_state.user, "question", prompt[:200])
            selected_chunks = retrieve_chunks(index, chunk_records, prompt)
            answer = answer_with_context(prompt, selected_chunks)

            with st.chat_message("assistant"):
                st.markdown(answer)

                st.markdown("### Sources")
                shown = set()

                for ch in selected_chunks:
                    label = f"{ch['source']} — {ch['location']} — chunk {ch['chunk_id']}"
                    if label not in shown:
                        st.markdown(f"- {label}")
                        shown.add(label)

                with st.expander("Retrieved Context"):
                    for i, ch in enumerate(selected_chunks, start=1):
                        st.markdown(f"**[{i}] {ch['source']} — {ch['location']} — chunk {ch['chunk_id']}**")
                        st.write(ch["text"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"An error occurred during retrieval or answering: {e}")
            log_event(st.session_state.user, "question_failed", str(e))

# Reports and suggestions
st.subheader("3) Management Insights")

col_r1, col_r2 = st.columns(2)
col_r3, col_r4 = st.columns(2)

if col_r1.button("Generate Report", use_container_width=True):
    if not chunk_records:
        st.error("You need to index documents first.")
    else:
        with st.spinner("Generating report..."):
            try:
                log_event(st.session_state.user, "generate_report", client_id)
                report = generate_report_from_chunks(chunk_records)
                st.markdown("### Report")
                st.write(report)
            except Exception as e:
                st.error(f"Failed to generate report: {e}")
                log_event(st.session_state.user, "generate_report_failed", str(e))

if col_r2.button("Generate Suggestions", use_container_width=True):
    if not chunk_records:
        st.error("You need to index documents first.")
    else:
        with st.spinner("Generating suggestions..."):
            try:
                log_event(st.session_state.user, "generate_suggestions", client_id)
                suggestions = generate_suggestions_from_chunks(chunk_records)
                st.markdown("### Suggestions")
                st.write(suggestions)
            except Exception as e:
                st.error(f"Failed to generate suggestions: {e}")
                log_event(st.session_state.user, "generate_suggestions_failed", str(e))

if col_r3.button("Generate Risk Highlights", use_container_width=True):
    if not chunk_records:
        st.error("You need to index documents first.")
    else:
        with st.spinner("Extracting risks..."):
            try:
                log_event(st.session_state.user, "generate_risks", client_id)
                risks = generate_risk_highlights(chunk_records)
                st.markdown("### Risk Highlights")
                st.write(risks)
            except Exception as e:
                st.error(f"Failed to extract risks: {e}")
                log_event(st.session_state.user, "generate_risks_failed", str(e))

if col_r4.button("Generate FAQ", use_container_width=True):
    if not chunk_records:
        st.error("You need to index documents first.")
    else:
        with st.spinner("Generating FAQ..."):
            try:
                log_event(st.session_state.user, "generate_faq", client_id)
                faq = generate_faq_from_chunks(chunk_records)
                st.markdown("### FAQ")
                st.write(faq)
            except Exception as e:
                st.error(f"Failed to generate FAQ: {e}")
                log_event(st.session_state.user, "generate_faq_failed", str(e))
