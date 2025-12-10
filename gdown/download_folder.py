# ============================================================
# gdown.download_folder
#
# Patched version with:
#   - Slow, retrying folder scans (SCAN_RETRY_COUNT + backoff + jitter)
#   - Slow, retrying downloads (FILE_RETRY_COUNT + backoff + jitter)
#   - Skip .git folders
#   - Manifest + resume support
#   - ThreadPoolExecutor workers
#   - Backwards-compatible _get_session() call (no verify kwarg issues)
# ============================================================

import collections
import itertools
import json
import os
import os.path as osp
import re
import sys
import time
import warnings
import random
from typing import Any, Dict, List, Optional, Union

from concurrent.futures import ThreadPoolExecutor, as_completed

import bs4

from .download import _get_session, download
from .exceptions import FolderContentsMaximumLimitError
from .parse_url import is_google_drive_url

# ------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ------------------------------------------------------------------
MAX_NUMBER_FILES = 1_000_000

# How many times to retry a *file download*
FILE_RETRY_COUNT = 20

# How many times to retry a *folder scan* (HTML / _DRIVE_ivd parse)
SCAN_RETRY_COUNT = 20

# Base waits in seconds
DOWNLOAD_BASE_SLEEP = 60    # base wait between download retries
SCAN_BASE_SLEEP = 60        # base wait between scan retries

# Caps in seconds (for exponential-ish backoff)
DOWNLOAD_MAX_SLEEP = 300    # cap download wait at 5 minutes
SCAN_MAX_SLEEP = 300        # cap scan wait at 5 minutes

# Jitter fraction (±20%)
JITTER_FRACTION = 0.2

GoogleDriveFileToDownload = collections.namedtuple(
    "GoogleDriveFileToDownload", ("id", "path", "local_path")
)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def _is_git_path(path: str) -> bool:
    """Return True if the path contains a .git directory."""
    parts = path.replace("\\", "/").split("/")
    return any(part == ".git" for part in parts)


def _sleep_with_backoff(
    base: int,
    attempt: int,
    max_sleep: int,
    what: str,
    quiet: bool = False,
) -> None:
    """
    Sleep with increasing backoff and random jitter.

    - base: base wait (e.g., 60s)
    - attempt: 1-based retry attempt index
    - max_sleep: cap in seconds
    - what: label for logging ("SCAN" or "DOWNLOAD")
    """
    # simple linear-ish backoff: base + (attempt-1)*30, capped
    wait = min(base + (attempt - 1) * 30, max_sleep)

    # apply ±20% jitter
    jitter_scale = 1.0 + random.uniform(-JITTER_FRACTION, JITTER_FRACTION)
    wait_jittered = max(5, int(wait * jitter_scale))  # never less than 5s

    if not quiet:
        print(
            f"[{what}] Backoff sleeping {wait_jittered}s (base={base}, "
            f"attempt={attempt}, raw_wait={wait})",
            file=sys.stderr,
        )

    time.sleep(wait_jittered)


class _GoogleDriveFile:
    TYPE_FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, id, name, type, children=None):
        self.id = id
        self.name = name
        self.type = type
        self.children = children if children is not None else []

    def is_folder(self):
        return self.type == self.TYPE_FOLDER


# ------------------------------------------------------------------
# Parsing the folder HTML
# ------------------------------------------------------------------

def _parse_google_drive_file(url: str, content: str):
    """Parse folder HTML and extract child files/folders."""
    soup = bs4.BeautifulSoup(content, features="html.parser")

    encoded_data = None
    for script in soup.select("script"):
        inner_html = script.decode_contents()
        if "_DRIVE_ivd" in inner_html:
            regex_iter = re.compile(r"'((?:[^'\\]|\\.)*)'").finditer(inner_html)
            try:
                encoded_data = next(itertools.islice(regex_iter, 1, None)).group(1)
            except StopIteration:
                raise RuntimeError("Could not find encoded folder metadata")
            break

    if encoded_data is None:
        raise RuntimeError(
            "Cannot retrieve folder metadata (Drive returned incomplete page)."
        )

    decoded = encoded_data.encode("utf-8").decode("unicode_escape")
    arr = json.loads(decoded)

    folder_contents = arr[0] if arr and arr[0] else []

    title = soup.title.contents[0]
    if " - " in title:
        name = title.rsplit(" - ", 1)[0]
    else:
        name = "folder"

    root = _GoogleDriveFile(
        id=url.split("/")[-1],
        name=name,
        type=_GoogleDriveFile.TYPE_FOLDER,
    )

    id_name_type_iter = [
        (e[0], e[2].encode("raw_unicode_escape").decode("utf-8"), e[3])
        for e in folder_contents
    ]
    return root, id_name_type_iter


def _scan_folder_once(
    sess,
    url: str,
    quiet: bool,
    remaining_ok: bool,
    verify: Union[bool, str],
):
    """
    Perform a single HTTP+parse attempt for a given folder URL.

    Raises on parse errors; caller is responsible for retry/backoff.
    """
    # Ensure English language to keep HTML more consistent
    if "?" in url:
        url_req = url + "&hl=en"
    else:
        url_req = url + "?hl=en"

    res = sess.get(url_req, verify=verify)
    if res.status_code != 200:
        raise RuntimeError(f"HTTP {res.status_code} while fetching {url_req}")

    root, entries = _parse_google_drive_file(url_req, res.text)

    if len(entries) == MAX_NUMBER_FILES and not remaining_ok:
        raise FolderContentsMaximumLimitError(
            "Folder contains more than permitted MAX_NUMBER_FILES"
        )

    return root, entries


def _download_and_parse_google_drive_link(
    sess,
    url: str,
    quiet: bool = False,
    remaining_ok: bool = False,
    verify: Union[bool, str] = True,
    _depth: int = 0,
):
    """
    Recursively scan Google Drive folder structure with retries and backoff.

    Returns (success: bool, root: _GoogleDriveFile or None)
    """

    indent = "  " * _depth
    if not quiet:
        print(f"{indent}[SCAN] Fetching folder: {url}", file=sys.stderr)

    # Retry loop around the HTML/parse step
    attempt = 0
    while True:
        attempt += 1
        try:
            root, entries = _scan_folder_once(sess, url, quiet, remaining_ok, verify)
            break
        except Exception as e:
            if attempt >= SCAN_RETRY_COUNT:
                if not quiet:
                    print(
                        f"{indent}[SCAN] Giving up after {attempt} attempts: {e}",
                        file=sys.stderr,
                    )
                return False, None

            if not quiet:
                print(
                    f"{indent}[SCAN] Error scanning folder (attempt "
                    f"{attempt}/{SCAN_RETRY_COUNT}): {e}",
                    file=sys.stderr,
                )

            _sleep_with_backoff(
                base=SCAN_BASE_SLEEP,
                attempt=attempt,
                max_sleep=SCAN_MAX_SLEEP,
                what="SCAN",
                quiet=quiet,
            )

    # Recurse into subfolders (each call has its own retry+backoff)
    for child_id, child_name, child_type in entries:
        if child_type != _GoogleDriveFile.TYPE_FOLDER:
            root.children.append(
                _GoogleDriveFile(id=child_id, name=child_name, type=child_type)
            )
            continue

        sub_url = "https://drive.google.com/drive/folders/" + child_id
        if not quiet:
            print(f"{indent}[SCAN] Entering subfolder: {child_name}", file=sys.stderr)

        ok, child = _download_and_parse_google_drive_link(
            sess=sess,
            url=sub_url,
            quiet=quiet,
            remaining_ok=remaining_ok,
            verify=verify,
            _depth=_depth + 1,
        )

        if not ok:
            return False, None

        root.children.append(child)

    return True, root


# ------------------------------------------------------------------
# Flattening directory tree
# ------------------------------------------------------------------

def _get_directory_structure(root: _GoogleDriveFile, base: str = ""):
    """Flatten a GoogleDriveFile tree into (id, path) pairs."""
    out = []
    for child in root.children:
        safe_name = child.name.replace(os.sep, "_")
        new_path = osp.join(base, safe_name)

        if child.is_folder():
            out.append((None, new_path))
            out.extend(_get_directory_structure(child, new_path))
        else:
            out.append((child.id, new_path))

    return out


# ------------------------------------------------------------------
# Manifest helpers
# ------------------------------------------------------------------

def _load_manifest(path: Optional[str]) -> Dict[str, Any]:
    if not path or not osp.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------------
# Backwards-compatible _get_session wrapper
# ------------------------------------------------------------------

def _make_session(
    proxy: Optional[str],
    use_cookies: bool,
    user_agent: Optional[str],
    verify: Union[bool, str],
):
    """
    Call _get_session() in a way that's compatible with multiple gdown versions.

    Newer gdown: _get_session(proxy=None, use_cookies=True, user_agent=None, verify=True)
    Older gdown: different subsets of those args, no 'verify' kwarg.
    """
    # Try newest signature first
    try:
        return _get_session(
            proxy=proxy,
            use_cookies=use_cookies,
            user_agent=user_agent,
            verify=verify,
        )
    except TypeError:
        pass

    # Try without verify
    try:
        return _get_session(
            proxy=proxy,
            use_cookies=use_cookies,
            user_agent=user_agent,
        )
    except TypeError:
        pass

    # Try only proxy + cookies
    try:
        return _get_session(
            proxy=proxy,
            use_cookies=use_cookies,
        )
    except TypeError:
        pass

    # Last resort: assume no args
    return _get_session()


# ------------------------------------------------------------------
# Main download_folder
# ------------------------------------------------------------------

def download_folder(
    url: Optional[str] = None,
    id: Optional[str] = None,
    output: Optional[str] = None,
    quiet: bool = False,
    proxy: Optional[str] = None,
    speed: Optional[float] = None,
    use_cookies: bool = True,
    remaining_ok: bool = False,
    verify: Union[bool, str] = True,
    user_agent: Optional[str] = None,
    skip_download: bool = False,
    resume: bool = False,
    manifest_path: Optional[str] = None,
    workers: Union[None, int, str] = None,
) -> Union[List[str], List[GoogleDriveFileToDownload], None]:
    """Download or enumerate entire Google Drive folder."""

    if not (url or id):
        raise ValueError("Either 'url' or 'id' must be provided")

    # Normalize URL
    if id and not url:
        url = f"https://drive.google.com/drive/folders/{id}"

    if user_agent is None:
        user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        )

    # Use compatibility wrapper instead of calling _get_session directly
    sess = _make_session(
        proxy=proxy,
        use_cookies=use_cookies,
        user_agent=user_agent,
        verify=verify,
    )

    if not quiet:
        print("[SCAN] Starting folder enumeration", file=sys.stderr)

    ok, root = _download_and_parse_google_drive_link(
        sess=sess,
        url=url,
        quiet=quiet,
        remaining_ok=remaining_ok,
        verify=verify,
    )

    if not ok or root is None:
        if not quiet:
            print("[SCAN] Failed to retrieve folder structure.", file=sys.stderr)
        return None

    structure = _get_directory_structure(root)

    # Skip .git paths
    structure = [(fid, p) for fid, p in structure if not _is_git_path(p)]

    # Listing-only mode
    if skip_download:
        base_out = output or os.getcwd()
        listed: List[GoogleDriveFileToDownload] = []
        for fid, path in structure:
            if fid is None:
                continue
            local_path = osp.join(base_out, root.name, path)
            listed.append(GoogleDriveFileToDownload(fid, path, local_path))
        if not quiet:
            print(f"[LIST] {len(listed)} files (skip_download=True)", file=sys.stderr)
        return listed

    # Build download tasks
    base_out = output or os.getcwd()
    root_out = osp.join(base_out, root.name)

    tasks: List[Dict[str, Any]] = []
    existing_paths: List[str] = []

    for fid, rel_path in structure:
        local_path = osp.join(root_out, rel_path)

        if fid is None:
            os.makedirs(local_path, exist_ok=True)
            continue

        if resume and osp.exists(local_path):
            if not quiet:
                print(f"[RESUME] Skipping existing file: {local_path}", file=sys.stderr)
            existing_paths.append(local_path)
            continue

        os.makedirs(osp.dirname(local_path), exist_ok=True)

        tasks.append(
            dict(
                id=fid,
                path=rel_path,
                local_path=local_path,
                url=f"https://drive.google.com/uc?id={fid}",
            )
        )

    if not tasks and not existing_paths:
        if not quiet:
            print("[DL] No files to download.", file=sys.stderr)
        return None

    # Decide worker count
    if workers in (None, 1):
        max_workers = None
    elif workers == "auto":
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu * 2)
    elif isinstance(workers, int) and workers > 1:
        max_workers = workers
    else:
        raise ValueError("workers must be None, 1, 'auto', or an int > 1")

    # Manifest
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("files", [])

    # ------------------------------------------------------------------
    # Download worker with backoff & jitter
    # ------------------------------------------------------------------
    def _download_one(task: Dict[str, Any]) -> Dict[str, Any]:
        fid = task["id"]
        rel = task["path"]
        local_path = task["local_path"]
        url_dl = task["url"]

        attempt = 0
        while True:
            attempt += 1
            try:
                if not quiet:
                    print(f"[DL] {rel} -> {local_path}", file=sys.stderr)

                res = download(
                    url=url_dl,
                    output=local_path,
                    quiet=quiet,
                    proxy=proxy,
                    speed=speed,
                    use_cookies=use_cookies,
                    verify=verify,
                    resume=resume,
                    user_agent=user_agent,
                    session=sess,
                )

                if res is None:
                    raise RuntimeError("Drive returned empty response")

                return dict(
                    id=fid,
                    path=rel,
                    local_path=local_path,
                    status="ok",
                )

            except Exception as e:
                if attempt >= FILE_RETRY_COUNT:
                    if not quiet:
                        print(
                            f"[DL] Giving up on {rel} after {attempt} attempts: {e}",
                            file=sys.stderr,
                        )
                    return dict(
                        id=fid,
                        path=rel,
                        local_path=local_path,
                        status=f"error: {e}",
                    )

                if not quiet:
                    print(
                        f"[DL] Error downloading {rel} (attempt "
                        f"{attempt}/{FILE_RETRY_COUNT}): {e}",
                        file=sys.stderr,
                    )

                _sleep_with_backoff(
                    base=DOWNLOAD_BASE_SLEEP,
                    attempt=attempt,
                    max_sleep=DOWNLOAD_MAX_SLEEP,
                    what="DOWNLOAD",
                    quiet=quiet,
                )

    # ------------------------------------------------------------------
    # Execute downloads
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []

    if max_workers is None:
        for t in tasks:
            results.append(_download_one(t))
    else:
        if not quiet:
            print(
                f"[DL] Downloading {len(tasks)} files with {max_workers} workers",
                file=sys.stderr,
            )
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            future_map = {exe.submit(_download_one, t): t for t in tasks}
            for fut in as_completed(future_map):
                results.append(fut.result())

    # Update manifest
    if manifest_path:
        index: Dict[str, Dict[str, Any]] = {}
        for entry in manifest.get("files", []):
            key = f"{entry.get('id')}::{entry.get('path')}"
            index[key] = entry

        for r in results:
            key = f"{r['id']}::{r['path']}"
            index[key] = dict(
                id=r["id"],
                path=r["path"],
                local_path=r["local_path"],
                status=r["status"],
            )

        manifest["files"] = list(index.values())
        _save_manifest(manifest_path, manifest)

    # Collect OK + resumed paths
    local_paths = list(existing_paths)
    local_paths.extend(
        [r["local_path"] for r in results if r.get("status") == "ok"]
    )

    if not local_paths:
        return None
    if len(local_paths) == 1:
        return local_paths[0]
    return local_paths