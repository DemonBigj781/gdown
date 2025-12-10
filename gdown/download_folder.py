# ============================================================
# gdown.download_folder
# Patched version with:
#   - Slow scanning (SCAN_SLEEP) to avoid rate limits
#   - Slow download retries (60 seconds)
#   - Skip .git folders
#   - Manifest + resume support
#   - ThreadPoolExecutor workers
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
FILE_RETRY_COUNT = 20
DOWNLOAD_RETRY_SLEEP = 60    # wait 60s between download retries
SCAN_SLEEP = 2               # wait 2 seconds between folder scans

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
        type=_GoogleDriveFile.TYPE_FOLDER
    )

    id_name_type_iter = [
        (e[0], e[2].encode("raw_unicode_escape").decode("utf-8"), e[3])
        for e in folder_contents
    ]
    return root, id_name_type_iter


def _download_and_parse_google_drive_link(
    sess, url, quiet=False, remaining_ok=False, verify=True
):
    """Recursively scan Google Drive folder structure."""

    # Slow scan to avoid Google rate-limits
    time.sleep(SCAN_SLEEP)

    if "?" in url:
        url = url + "&hl=en"
    else:
        url = url + "?hl=en"

    res = sess.get(url, verify=verify)
    if res.status_code != 200:
        return False, None

    root, entries = _parse_google_drive_file(url, res.text)

    # For each child:
    for child_id, child_name, child_type in entries:

        if child_type != _GoogleDriveFile.TYPE_FOLDER:
            root.children.append(
                _GoogleDriveFile(id=child_id, name=child_name, type=child_type)
            )
            continue

        # It is a subfolder — recurse slowly
        if not quiet:
            print(f"Scanning subfolder {child_name}...", file=sys.stderr)

        ok, child = _download_and_parse_google_drive_link(
            sess,
            "https://drive.google.com/drive/folders/" + child_id,
            quiet=quiet,
            remaining_ok=remaining_ok,
            verify=verify,
        )

        if not ok:
            return False, None

        root.children.append(child)

    if len(root.children) == MAX_NUMBER_FILES and not remaining_ok:
        raise FolderContentsMaximumLimitError(
            "Folder contains more than permitted MAX_NUMBER_FILES"
        )

    return True, root


# ------------------------------------------------------------------
# Flattening directory tree
# ------------------------------------------------------------------

def _get_directory_structure(root: _GoogleDriveFile, base=""):
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

def _load_manifest(path: str):
    if not path or not osp.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(path: str, data: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------------
# Main download_folder
# ------------------------------------------------------------------

def download_folder(
    url=None,
    id=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    remaining_ok=False,
    verify=True,
    user_agent=None,
    skip_download=False,
    resume=False,
    manifest_path=None,
    workers=None,
):
    """Download or enumerate entire Google Drive folder."""

    if not (url or id):
        raise ValueError("Either 'url' or 'id' must be provided")

    # Normalize URL
    if id:
        url = f"https://drive.google.com/drive/folders/{id}"

    if user_agent is None:
        user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        )

    sess = _get_session(
        proxy=proxy,
        use_cookies=use_cookies,
        user_agent=user_agent,
        verify=verify,
    )

    if not quiet:
        print("Scanning folder structure...", file=sys.stderr)

    ok, root = _download_and_parse_google_drive_link(
        sess,
        url,
        quiet=quiet,
        remaining_ok=remaining_ok,
        verify=verify,
    )

    if not ok:
        print("Failed to retrieve folder structure.", file=sys.stderr)
        return None

    structure = _get_directory_structure(root)

    # Skip .git paths
    structure = [(fid, p) for fid, p in structure if not _is_git_path(p)]

    # Only listing tasks
    if skip_download:
        out = []
        for fid, path in structure:
            if fid is None:
                continue
            lp = osp.join(output or os.getcwd(), path)
            out.append(GoogleDriveFileToDownload(fid, path, lp))
        return out

    # Build download tasks
    output = output or os.getcwd()
    root_out = osp.join(output, root.name)

    tasks = []
    for fid, rel_path in structure:
        local_path = osp.join(root_out, rel_path)

        if fid is None:
            os.makedirs(local_path, exist_ok=True)
            continue

        # Resume check
        if resume and osp.exists(local_path):
            if not quiet:
                print(f"Skipping existing file: {local_path}", file=sys.stderr)
            continue

        os.makedirs(osp.dirname(local_path), exist_ok=True)

        tasks.append(dict(
            id=fid,
            path=rel_path,
            local_path=local_path,
            url=f"https://drive.google.com/uc?id={fid}",
        ))

    # Worker decision
    if workers in (None, 1):
        max_workers = None
    elif workers == "auto":
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu * 2)
    else:
        max_workers = int(workers)

    # Download worker ---------------------------------------

    def _download_one(task):
        fid = task["id"]
        rel = task["path"]
        lp = task["local_path"]
        url_dl = task["url"]

        attempt = 0
        while True:
            attempt += 1
            try:
                if not quiet:
                    print(f"Downloading {rel}...", file=sys.stderr)

                res = download(
                    url=url_dl,
                    output=lp,
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

                return dict(id=fid, path=rel, local_path=lp, status="ok")

            except Exception as e:
                if attempt >= FILE_RETRY_COUNT:
                    return dict(id=fid, path=rel, local_path=lp, status=f"error: {e}")

                if not quiet:
                    print(
                        f"Error downloading {rel}: {e} "
                        f"(retry {attempt}/{FILE_RETRY_COUNT}) "
                        f"Waiting {DOWNLOAD_RETRY_SLEEP}s...",
                        file=sys.stderr,
                    )
                time.sleep(DOWNLOAD_RETRY_SLEEP)

    # Execute tasks ------------------------------------------

    results = []

    if max_workers is None:
        for t in tasks:
            results.append(_download_one(t))
    else:
        if not quiet:
            print(f"Downloading {len(tasks)} files with {max_workers} workers", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_download_one, t): t for t in tasks}
            for f in as_completed(futures):
                results.append(f.result())

    # Collect OK results
    ok_paths = [r["local_path"] for r in results if r["status"] == "ok"]

    if not ok_paths:
        return None
    if len(ok_paths) == 1:
        return ok_paths[0]
    return ok_paths