"""
YouTube API Comment Bot – Server
=================================
Runs on Render (or any host). Receives job instructions from the local GUI,
uses the YouTube Data API v3 to search for videos, generates AI comments via
OpenRouter / Cerebras, and posts them one-by-one respecting time limits.

OAuth2 tokens are sent by the local GUI (which runs the consent screen flow)
and stored in-memory. The server can refresh expired access tokens using the
stored refresh token + client credentials from environment variables.
"""

import json
import os
import time
import random
import threading
import uuid
import logging
from datetime import datetime, timezone
from collections import deque

import requests as http_requests
from flask import Flask, request, jsonify, render_template

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("youtube-api-server")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
YOUR_SITE_URL = "AIBrainL.ink"
YOUR_APP_NAME = "YouTube API Commenter"

PERSONAS = {
    "teenager": "Respond as a texting teenager with lots of spelling mistakes, grammatical errors, run-on sentences, capitalization issues, and punctuation problems.",
    "normal": "Respond as a normal YouTube commenter, with casual language and occasional spelling mistakes or grammatical errors.",
    "educated": "Respond as an educated person with very rare spelling mistakes, providing thoughtful and well-structured comments.",
    "bot": "Respond with perfect spelling, grammar, and punctuation, like a bot would.",
}

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
server_start_time = datetime.now(timezone.utc)

# OAuth credentials (set by GUI via /api/set-auth)
oauth_lock = threading.Lock()
oauth_credentials: Credentials | None = None

# Client credentials for token refresh (from env vars or credentials.json)
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
TOKEN_URI = "https://oauth2.googleapis.com/token"

# Try loading from credentials.json if env vars not set
_creds_file = os.environ.get(
    "GOOGLE_CREDENTIALS_JSON",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "client_secret_25463113626-u8qk0sc70bma6fao420jgodq106irfgg.apps.googleusercontent.com.json")
)
if not CLIENT_ID and os.path.exists(_creds_file):
    try:
        with open(_creds_file, "r") as f:
            _cdata = json.load(f)
            _installed = _cdata.get("installed", _cdata.get("web", {}))
            CLIENT_ID = _installed.get("client_id", "")
            CLIENT_SECRET = _installed.get("client_secret", "")
            TOKEN_URI = _installed.get("token_uri", TOKEN_URI)
    except Exception as e:
        log.warning(f"Could not load credentials.json: {e}")

# Job queue & state
job_lock = threading.Lock()
current_job: dict | None = None
job_thread: threading.Thread | None = None
job_running = False
job_stop_event = threading.Event()

# Logs (in-memory ring buffer)
MAX_LOGS = 2000
log_buffer: list[dict] = []
log_buffer_lock = threading.Lock()

# Stats
stats_lock = threading.Lock()
total_videos_found = 0
total_ai_generated = 0
total_ai_failed = 0
total_comments_posted = 0
total_comments_failed = 0
total_api_calls = 0  # YouTube API search + comment calls

# Comment history
MAX_HISTORY = 500
comment_history: list[dict] = []
history_lock = threading.Lock()

# Currently processing
currently_processing: dict | None = None
currently_processing_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal logging helper
# ---------------------------------------------------------------------------
def _log(message: str, level: str = "info"):
    """Log to both Python logger and in-memory buffer."""
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"timestamp": ts, "level": level, "message": message}

    if level == "error":
        log.error(message)
    elif level == "warning":
        log.warning(message)
    else:
        log.info(message)

    with log_buffer_lock:
        log_buffer.append(entry)
        if len(log_buffer) > MAX_LOGS:
            log_buffer.pop(0)


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------
def _get_youtube_service():
    """Return an authenticated YouTube Data API service, refreshing if needed."""
    global oauth_credentials

    with oauth_lock:
        creds = oauth_credentials

    if creds is None:
        return None

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        try:
            _log("Access token expired – refreshing...")
            creds.refresh(GoogleAuthRequest())
            with oauth_lock:
                oauth_credentials = creds
            _log("Access token refreshed successfully")
        except Exception as e:
            _log(f"Failed to refresh token: {e}", "error")
            return None

    if not creds.valid:
        _log("OAuth credentials are not valid and cannot be refreshed", "error")
        return None

    try:
        with stats_lock:
            global total_api_calls
            total_api_calls += 1
        return build("youtube", "v3", credentials=creds)
    except Exception as e:
        _log(f"Failed to build YouTube service: {e}", "error")
        return None


def _is_auth_valid() -> bool:
    """Check if we have valid (or refreshable) OAuth credentials."""
    global oauth_credentials
    with oauth_lock:
        creds = oauth_credentials
    if creds is None:
        return False
    if creds.valid:
        return True
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            with oauth_lock:
                oauth_credentials = creds
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# AI Comment Generation (same logic as youtube_commenter.py)
# ---------------------------------------------------------------------------
def generate_ai_comment(title, persona, ai_response_length=0, openrouter_api_key=None,
                        custom_model=None, custom_prompt=None,
                        website_address=None, model_to_use="OpenRouter",
                        cerebras_api_key=None, cerebras_model=None):
    _log(f"Generating AI comment for: {title[:80]}...")

    length_instruction = f"Generate a response that is approximately {ai_response_length} words long. " if ai_response_length > 0 else ""

    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.format(
            title=title,
            length=length_instruction,
            website=website_address or ""
        )
    else:
        prompt = f"{PERSONAS.get(persona, PERSONAS['normal'])} {length_instruction}Based on the following YouTube video title, generate an appropriate and insightful comment response. "
        if website_address:
            prompt += f"Include this website in your response: {website_address}. "
        prompt += f"\n\nVideo Title: {title}\n"

    prompt += "\nGenerated comment:"

    with stats_lock:
        global total_api_calls
        total_api_calls += 1

    if model_to_use == "Cerebras":
        return _generate_cerebras_comment(prompt, cerebras_api_key, cerebras_model)
    else:
        return _generate_openrouter_comment(prompt, openrouter_api_key, custom_model)


def _generate_cerebras_comment(prompt, cerebras_api_key, cerebras_model):
    try:
        if not cerebras_api_key or not cerebras_api_key.strip():
            _log("Cerebras API key not set", "error")
            return None

        model = cerebras_model.strip() if cerebras_model and cerebras_model.strip() else "llama-3.3-70b"
        _log(f"Using Cerebras model: {model}")

        headers = {
            "Authorization": f"Bearer {cerebras_api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        url = "https://api.cerebras.ai/v1/preview/chat/completions" if "preview" in model else "https://api.cerebras.ai/v1/chat/completions"

        response = http_requests.post(url=url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        ai_comment = response.json()["choices"][0]["message"]["content"]
        _log(f"AI comment generated (Cerebras): {ai_comment[:80]}...")
        return ai_comment
    except Exception as e:
        _log(f"Cerebras AI error: {e}", "error")
        return None


def _generate_openrouter_comment(prompt, openrouter_api_key, custom_model):
    try:
        api_key = openrouter_api_key.strip() if openrouter_api_key and openrouter_api_key.strip() else ""

        if not api_key:
            _log("OpenRouter API key not set", "error")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_APP_NAME,
        }
        payload = {
            "model": custom_model or "google/gemma-2-9b-it:free",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = http_requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        ai_comment = response.json()["choices"][0]["message"]["content"]
        _log(f"AI comment generated (OpenRouter): {ai_comment[:80]}...")
        return ai_comment
    except Exception as e:
        _log(f"OpenRouter AI error: {e}", "error")
        return None


# ---------------------------------------------------------------------------
# YouTube API: Search videos
# ---------------------------------------------------------------------------
def search_videos(youtube, keyword, max_results=10, sort_order="relevance"):
    """Search YouTube for videos matching a keyword. Returns list of video info dicts."""
    _log(f"Searching YouTube API for: '{keyword}' (max={max_results}, order={sort_order})")

    order_map = {
        "Relevance": "relevance",
        "Upload date": "date",
        "View count": "viewCount",
        "Rating": "rating",
    }
    api_order = order_map.get(sort_order, "relevance")

    videos = []
    next_page = None
    remaining = max_results

    try:
        while remaining > 0:
            batch_size = min(remaining, 50)  # API max per page is 50

            with stats_lock:
                global total_api_calls
                total_api_calls += 1

            req = youtube.search().list(
                part="snippet",
                q=keyword,
                type="video",
                maxResults=batch_size,
                order=api_order,
                pageToken=next_page,
            )
            response = req.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]
                video_id = item["id"]["videoId"]
                videos.append({
                    "search_keyword": keyword,
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", "Unknown"),
                    "description_snippet": snippet.get("description", "")[:200],
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "ai_comment": None,
                })

            remaining -= batch_size
            next_page = response.get("nextPageToken")
            if not next_page:
                break

        _log(f"Found {len(videos)} videos for '{keyword}'")

        with stats_lock:
            global total_videos_found
            total_videos_found += len(videos)

    except Exception as e:
        _log(f"YouTube search API error: {e}", "error")

    return videos[:max_results]


# ---------------------------------------------------------------------------
# YouTube API: Post comment
# ---------------------------------------------------------------------------
def post_comment(youtube, video_id, comment_text):
    """Post a top-level comment on a YouTube video using the Data API."""
    _log(f"Posting comment to video {video_id}...")

    try:
        with stats_lock:
            global total_api_calls
            total_api_calls += 1

        body = {
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": comment_text
                    }
                }
            }
        }

        response = youtube.commentThreads().insert(
            part="snippet",
            body=body,
        ).execute()

        comment_id = response.get("id", "")
        _log(f"Comment posted successfully (id={comment_id}) on video {video_id}")

        with stats_lock:
            global total_comments_posted
            total_comments_posted += 1

        return True, comment_id

    except Exception as e:
        error_msg = str(e)
        _log(f"Failed to post comment on {video_id}: {error_msg}", "error")

        with stats_lock:
            global total_comments_failed
            total_comments_failed += 1

        return False, error_msg


# ---------------------------------------------------------------------------
# Background job worker
# ---------------------------------------------------------------------------
def _job_worker(params: dict):
    """Main job loop: search → generate AI comments → post them."""
    global job_running, current_job, currently_processing

    _log("=" * 60)
    _log("  JOB STARTED")
    _log("=" * 60)

    try:
        # Extract parameters
        search_keywords = params.get("search_keywords", [])
        sort_filter = params.get("sort_filter", "Relevance")
        max_videos = params.get("max_videos", 10)
        min_wait = params.get("min_wait_time", 30)
        max_wait = params.get("max_wait_time", 120)
        ai_response_length = params.get("ai_response_length", 0)
        ai_delay = params.get("ai_delay_between_responses", 5)
        persona = params.get("persona", "normal")
        openrouter_api_key = params.get("openrouter_api_key", "")
        custom_model = params.get("custom_model", "")
        custom_prompt = params.get("custom_prompt", "")
        website_address = params.get("website_address", "")
        model_to_use = params.get("model_to_use", "OpenRouter")
        cerebras_api_key = params.get("cerebras_api_key", "")
        cerebras_model = params.get("cerebras_model", "")

        _log(f"Keywords: {search_keywords}")
        _log(f"Sort: {sort_filter} | Max videos/keyword: {max_videos}")
        _log(f"Wait between posts: {min_wait}-{max_wait}s")
        _log(f"AI model: {model_to_use} | Persona: {persona}")

        # Get YouTube service
        youtube = _get_youtube_service()
        if youtube is None:
            _log("Cannot proceed – no valid YouTube API credentials", "error")
            return

        # ========== PHASE 1: SEARCH ==========
        _log("")
        _log("=" * 60)
        _log("  PHASE 1: SEARCHING VIDEOS")
        _log("=" * 60)

        all_videos = []
        for keyword in search_keywords:
            if job_stop_event.is_set():
                _log("Job stopped by user during search phase")
                return

            videos = search_videos(youtube, keyword, max_videos, sort_filter)
            all_videos.extend(videos)
            _log(f"Keyword '{keyword}': {len(videos)} videos found")

        total = len(all_videos)
        _log(f"\nTotal videos found across all keywords: {total}")

        if total == 0:
            _log("No videos found. Job complete.")
            return

        # ========== PHASE 2: AI COMMENT GENERATION ==========
        _log("")
        _log("=" * 60)
        _log("  PHASE 2: GENERATING AI COMMENTS")
        _log("=" * 60)

        for idx, video in enumerate(all_videos):
            if job_stop_event.is_set():
                _log("Job stopped by user during AI generation phase")
                return

            with currently_processing_lock:
                currently_processing = {
                    "phase": "ai_generation",
                    "index": idx + 1,
                    "total": total,
                    "video": video["title"][:80],
                }

            _log(f"\n--- AI Generation [{idx + 1}/{total}] ---")
            _log(f"  Video: {video['title'][:80]}")

            try:
                ai_comment = generate_ai_comment(
                    video["title"],
                    persona,
                    ai_response_length,
                    openrouter_api_key,
                    custom_model,
                    custom_prompt,
                    website_address,
                    model_to_use,
                    cerebras_api_key,
                    cerebras_model,
                )
                video["ai_comment"] = ai_comment

                if ai_comment:
                    with stats_lock:
                        global total_ai_generated
                        total_ai_generated += 1
                    _log(f"  [OK] Comment generated ({len(ai_comment)} chars)")
                else:
                    with stats_lock:
                        global total_ai_failed
                        total_ai_failed += 1
                    _log(f"  [FAIL] Comment generation failed")
            except Exception as e:
                with stats_lock:
                    total_ai_failed += 1
                _log(f"  [FAIL] AI error: {e}", "error")
                video["ai_comment"] = None

            if idx < total - 1:
                _log(f"  Waiting {ai_delay}s before next AI request...")
                for _ in range(int(ai_delay * 2)):
                    if job_stop_event.is_set():
                        return
                    time.sleep(0.5)

        _log(f"\nAI generation complete for {total} videos.")

        # ========== PHASE 3: POSTING COMMENTS ==========
        _log("")
        _log("=" * 60)
        _log("  PHASE 3: POSTING COMMENTS")
        _log("=" * 60)

        # Re-acquire YouTube service (token might have refreshed)
        youtube = _get_youtube_service()
        if youtube is None:
            _log("Cannot post – YouTube API credentials lost/expired", "error")
            return

        posted = 0
        failed = 0
        is_first = True

        for idx, video in enumerate(all_videos):
            if job_stop_event.is_set():
                _log("Job stopped by user during posting phase")
                break

            ai_comment = video.get("ai_comment")
            if not ai_comment:
                _log(f"[SKIP] No AI comment for: {video['title'][:60]}")
                failed += 1
                continue

            with currently_processing_lock:
                currently_processing = {
                    "phase": "posting",
                    "index": idx + 1,
                    "total": total,
                    "video": video["title"][:80],
                }

            # Wait between posts
            if not is_first:
                delay = random.uniform(min_wait, max_wait)
                _log(f"Waiting {delay:.1f}s before next post...")
                for _ in range(int(delay * 2)):
                    if job_stop_event.is_set():
                        _log("Job stopped during wait")
                        break
                    time.sleep(0.5)
                if job_stop_event.is_set():
                    break
            is_first = False

            _log(f"\n--- Posting [{idx + 1}/{total}] ---")
            _log(f"  Video: {video['title'][:80]}")
            _log(f"  Channel: {video['channel']}")
            _log(f"  Comment: {ai_comment[:120]}...")

            success, result_info = post_comment(youtube, video["video_id"], ai_comment)

            # Record in history
            record = {
                "id": str(uuid.uuid4())[:8],
                "video_id": video["video_id"],
                "title": video["title"][:100],
                "channel": video["channel"],
                "keyword": video["search_keyword"],
                "comment": ai_comment[:200],
                "success": success,
                "result": result_info if not success else "",
                "posted_at": datetime.now(timezone.utc).isoformat(),
            }
            with history_lock:
                comment_history.append(record)
                if len(comment_history) > MAX_HISTORY:
                    comment_history.pop(0)

            if success:
                posted += 1
                _log(f"  [OK] Comment posted ({posted} total)")
            else:
                failed += 1
                _log(f"  [FAIL] {result_info}")

        _log("")
        _log("=" * 60)
        _log(f"  JOB COMPLETE: {posted} posted, {failed} failed, {total} total")
        _log("=" * 60)

    except Exception as e:
        _log(f"Job crashed: {e}", "error")
        import traceback
        _log(traceback.format_exc(), "error")
    finally:
        with currently_processing_lock:
            currently_processing = None
        with job_lock:
            global current_job
            job_running = False
            current_job = None
        job_stop_event.clear()


def _start_job(params: dict):
    """Launch the job worker thread."""
    global job_thread, job_running, current_job

    with job_lock:
        if job_running:
            return False, "A job is already running"

        job_running = True
        current_job = params
        job_stop_event.clear()

    job_thread = threading.Thread(target=_job_worker, args=(params,), daemon=True)
    job_thread.start()
    return True, "Job started"


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


@app.route("/api/check-auth", methods=["GET"])
def check_auth():
    """Check if OAuth token is valid / refreshable."""
    valid = _is_auth_valid()
    return jsonify({
        "auth_valid": valid,
        "client_id": CLIENT_ID[:20] + "..." if CLIENT_ID else "",
    })


@app.route("/api/set-auth", methods=["POST"])
def set_auth():
    """
    Receive OAuth2 credentials from the local GUI.

    Expected JSON body:
    {
        "access_token": "...",
        "refresh_token": "...",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "...",
        "client_secret": "...",
        "expiry": "2025-01-01T00:00:00Z"   (optional)
    }
    """
    global oauth_credentials

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    access_token = data.get("access_token", "")
    refresh_token = data.get("refresh_token", "")

    if not access_token:
        return jsonify({"error": "access_token is required"}), 400

    try:
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri=data.get("token_uri", TOKEN_URI),
            client_id=data.get("client_id", CLIENT_ID),
            client_secret=data.get("client_secret", CLIENT_SECRET),
            scopes=SCOPES,
        )

        with oauth_lock:
            oauth_credentials = creds

        _log("OAuth credentials received and stored from GUI")
        return jsonify({"message": "Auth credentials stored", "valid": creds.valid})

    except Exception as e:
        _log(f"Failed to set auth: {e}", "error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/start-job", methods=["POST"])
def start_job():
    """
    Start a YouTube comment job.

    Expected JSON body:
    {
        "search_keywords": ["keyword1", "keyword2"],
        "sort_filter": "Relevance",
        "max_videos": 10,
        "min_wait_time": 30,
        "max_wait_time": 120,
        "ai_response_length": 30,
        "persona": "normal",
        "openrouter_api_key": "...",
        "custom_model": "...",
        "custom_prompt": "...",
        "website_address": "...",
        "model_to_use": "OpenRouter",
        "cerebras_api_key": "...",
        "cerebras_model": "...",
        "ai_delay_between_responses": 5.0
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    # Check auth first
    if not _is_auth_valid():
        return jsonify({"error": "auth_required", "message": "YouTube OAuth token is missing or expired. Please authenticate first."}), 401

    keywords = data.get("search_keywords", [])
    if not keywords:
        return jsonify({"error": "No search_keywords provided"}), 400

    ok, msg = _start_job(data)
    if ok:
        _log(f"Job started with {len(keywords)} keyword(s)")
        return jsonify({"message": msg, "keywords": keywords}), 200
    else:
        return jsonify({"error": msg}), 409


@app.route("/api/stop-job", methods=["POST"])
def stop_job():
    """Stop the currently running job."""
    with job_lock:
        if not job_running:
            return jsonify({"message": "No job is running"}), 200

    job_stop_event.set()
    _log("Job stop requested by user")
    return jsonify({"message": "Stop signal sent"})


@app.route("/api/status", methods=["GET"])
def status():
    """Return current job status, queue, and recent history."""
    with job_lock:
        running = job_running
        job_params = current_job

    with currently_processing_lock:
        cp = currently_processing

    with history_lock:
        history = list(comment_history[-50:])

    with stats_lock:
        stats = {
            "total_videos_found": total_videos_found,
            "total_ai_generated": total_ai_generated,
            "total_ai_failed": total_ai_failed,
            "total_comments_posted": total_comments_posted,
            "total_comments_failed": total_comments_failed,
            "total_api_calls": total_api_calls,
        }

    return jsonify({
        "job_running": running,
        "currently_processing": cp,
        "stats": stats,
        "recent_history": history,
        "auth_valid": _is_auth_valid(),
    })


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Return recent logs. Optional ?limit=N query param."""
    limit = request.args.get("limit", 200, type=int)
    with log_buffer_lock:
        logs = list(log_buffer[-limit:])
    return jsonify({"logs": logs})


@app.route("/api/clear-logs", methods=["POST"])
def clear_logs():
    """Clear the in-memory log buffer."""
    with log_buffer_lock:
        log_buffer.clear()
    return jsonify({"message": "Logs cleared"})


@app.route("/api/keepalive", methods=["GET"])
def keepalive():
    return jsonify({
        "status": "alive",
        "time": datetime.now(timezone.utc).isoformat(),
        "job_running": job_running,
    })


@app.route("/api/dashboard-status", methods=["GET"])
def dashboard_status():
    """Aggregate status for the dashboard UI."""
    with job_lock:
        running = job_running

    with currently_processing_lock:
        cp = currently_processing

    with history_lock:
        history = list(comment_history[-50:])

    with stats_lock:
        stats = {
            "total_videos_found": total_videos_found,
            "total_ai_generated": total_ai_generated,
            "total_ai_failed": total_ai_failed,
            "total_comments_posted": total_comments_posted,
            "total_comments_failed": total_comments_failed,
            "total_api_calls": total_api_calls,
        }

    with log_buffer_lock:
        recent_logs = list(log_buffer[-100:])

    auth_valid = _is_auth_valid()

    # Uptime
    delta = datetime.now(timezone.utc) - server_start_time
    secs = int(delta.total_seconds())
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if days:  parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if mins:  parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    uptime = " ".join(parts)

    return jsonify({
        "job_running": running,
        "currently_processing": cp,
        "stats": stats,
        "recent_history": history,
        "recent_logs": recent_logs,
        "auth_valid": auth_valid,
        "uptime": uptime,
    })


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
