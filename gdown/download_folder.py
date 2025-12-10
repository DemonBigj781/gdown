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

MAX_NUMBER_FILES = 1000000
FILE_RETRY_COUNT = 5
FILE_RETRY_SLEEP = 10

GoogleDriveFileToDownload = collections.namedtuple(
    "GoogleDriveFileToDownload", ("id", "path", "local_path")
)


class _GoogleDriveFile(object):
    TYPE_FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, id: str, name: str, mime_type: str, children=None):
        self.id = id
        self.name = name
        self.mime_type = mime_type
        self.children: List["_GoogleDriveFile"] = children if children is not None else []

    def is_folder(self) -> bool:
        return self.mime_type == self.TYPE_FOLDER

    def __repr__(self) -> str:
        return "_GoogleDriveFile(id={!r}, name={!r}, mime_type={!r}, children={!r})".format(
            self.id, self.name, self.mime_type, self.children
        )


def _is_git_path(path: str) -> bool:
    """
    Return True if this logical path is inside a .git directory.

    We normalize separators and look for '.git' as a path segment,
    so things like '.git/config', 'folder/.git/HEAD', etc. are excluded.
    """
    parts = path.replace("\\", "/").split("/")
    return any(part == ".git" for part in parts)


def _id_to_folder_url(folder_id: str) -> str:
    return "https://drive.google.com/drive/folders/{}".format(folder_id)


def _id_to_download_url(file_id: str) -> str:
    # use uc endpoint so gdown.download can handle confirmation etc.
    return "https://drive.google.com/uc?id={}".format(file_id)


def _extract_encoded_drive_data(html: str) -> str:
    """
    Extract the encoded data string from the folder HTML.

    Google Drive folder pages include a JS snippet where a global variable
    like _DRIVE_ivd is assigned an encoded array/string describing contents.
    We locate the script tag, then pull out the second JS string literal.
    """
    soup = bs4.BeautifulSoup(html, features="html.parser")

    encoded_data = None
    for script in soup.select("script"):
        inner_html = script.decode_contents()
        if "_DRIVE_ivd" not in inner_html:
            continue

        # Find JS string literals inside the script code
        # The first string is usually the variable name, the second is the payload
        regex_iter = re.compile(r"'((?:[^'\\]|\\.)*)'").finditer(inner_html)
        try:
            encoded_data = next(itertools.islice(regex_iter, 1, None)).group(1)
        except StopIteration:
            raise RuntimeError("Couldn't find encoded folder metadata in Drive HTML.")
        break

    if encoded_data is None:
        raise RuntimeError(
            "Cannot retrieve the folder information from the link. "
            "You may need to change the permission to 'Anyone with the link', "
            "or there might be too many recent accesses."
        )

    # JS string literal uses escape sequences, convert to a proper Python string
    return encoded_data.encode("utf-8").decode("unicode_escape")


def _parse_google_drive_file(url: str, html: str) -> _GoogleDriveFile:
    """
    Parse a Google Drive folder HTML page into a root _GoogleDriveFile tree.

    This function reconstructs a directory tree from encoded metadata embedded
    in the page. The structure of the payload can change over time; this parser
    aims to be robust but may need updates if Google changes their format.
    """
    encoded = _extract_encoded_drive_data(html)

    # The encoded string is often JSON-like or contains a JSON array embedded.
    # We look for the first top-level JSON array as a starting point.
    # This is intentionally conservative to avoid brittle assumptions.
    match = re.search(r"(\[\[.*\]\])", encoded, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Could not locate folder metadata array in encoded payload.")

    try:
        data = json.loads(match.group(1))
    except Exception as e:
        raise RuntimeError("Failed to decode folder metadata JSON: {}".format(e))

    # We expect 'data' to contain rows describing items in the folder.
    # The exact indexing can vary. We look for entries that look like:
    # [ ..., file_id, ..., file_name, ..., mime_type, ... ]
    # and build a tree using parent references.
    #
    # This is a best-effort reconstruction, not a byte-for-byte reproduction
    # of any particular implementation.
    by_id: Dict[str, _GoogleDriveFile] = {}
    children_map: Dict[str, List[str]] = {}
    root_id: Optional[str] = None

    for row in data:
        if not isinstance(row, list):
            continue

        # Heuristic positions: id, name, mime_type, parent_id
        file_id = None
        name = None
        mime_type = None
        parent_id = None

        for item in row:
            # Very rough heuristics; actual indices depend on Google's internals.
            if isinstance(item, str):
                if not file_id and re.match(r"^[a-zA-Z0-9\-_]{10,}$", item):
                    file_id = item
                elif not name:
                    name = item
                elif not mime_type and item.startswith("application/"):
                    mime_type = item

        # Attempt to find parent id by looking at last long-ish string
        long_strings = [
            s for s in row if isinstance(s, str) and len(s) >= 10 and s != file_id
        ]
        if long_strings:
            parent_id = long_strings[-1]

        if file_id and name and mime_type:
            if file_id not in by_id:
                by_id[file_id] = _GoogleDriveFile(file_id, name, mime_type)

            if parent_id:
                children_map.setdefault(parent_id, []).append(file_id)

            # Heuristic: if this looks like the top-level folder (TYPE_FOLDER and matches id in URL)
            if mime_type == _GoogleDriveFile.TYPE_FOLDER and file_id in url:
                root_id = file_id

    if not by_id:
        raise RuntimeError("Parsed folder metadata is empty; Drive format may have changed.")

    # Build children lists
    for parent, child_ids in children_map.items():
        if parent not in by_id:
            continue
        parent_node = by_id[parent]
        for cid in child_ids:
            child_node = by_id.get(cid)
            if child_node:
                parent_node.children.append(child_node)

    if root_id is None:
        # Fallback: choose any folder-like node as root
        for file_id, node in by_id.items():
            if node.is_folder():
                root_id = file_id
                break

    if root_id is None or root_id not in by_id:
        raise RuntimeError("Could not determine root folder for Google Drive tree.")

    return by_id[root_id]


def _walk_drive_tree(root: _GoogleDriveFile, parent_path: str = "") -> List[GoogleDriveFileToDownload]:
    """
    Flatten a _GoogleDriveFile tree into a list of (id, path) entries.

    'path' is the logical path inside the folder (e.g., 'subdir/file.txt').
    """
    results: List[GoogleDriveFileToDownload] = []

    def _recurse(node: _GoogleDriveFile, current_path: str) -> None:
        if node.is_folder():
            folder_path = osp.join(current_path, node.name) if current_path else node.name
            for child in node.children:
                _recurse(child, folder_path)
        else:
            file_path = osp.join(current_path, node.name) if current_path else node.name
            results.append(
                GoogleDriveFileToDownload(
                    id=node.id,
                    path=file_path,
                    local_path=None,  # to be filled later
                )
            )

    _recurse(root, parent_path)
    return results


def _get_directory_structure(
    url: Optional[str] = None,
    id: Optional[str] = None,
    session=None,
    use_cookies: bool = True,
    verify: Union[bool, str] = True,
    user_agent: Optional[str] = None,
) -> _GoogleDriveFile:
    """
    Fetch and parse a Google Drive folder page into a _GoogleDriveFile tree.
    """
    if id is not None and url is None:
        url = _id_to_folder_url(id)

    if url is None:
        raise ValueError("Either url or id must be specified.")

    if not is_google_drive_url(url):
        raise ValueError("URL must be a Google Drive link: {}".format(url))

    if session is None:
        session = _get_session(proxy=None, use_cookies=use_cookies, verify=verify, user_agent=user_agent)

    response = session.get(url, stream=True)
    response.raise_for_status()
    html = response.text

    root = _parse_google_drive_file(url, html)
    return root


def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    if not osp.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt or unreadable manifest: ignore
        return {}


def _save_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_path, manifest_path)


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
) -> Union[None, str, List[str], List[GoogleDriveFileToDownload]]:
    """
    Download or enumerate all files inside a public Google Drive folder.

    Parameters
    ----------
    url, id:
        Google Drive folder URL or folder ID.
    output:
        Local directory where files will be stored.
    quiet:
        If True, suppress progress messages.
    proxy, speed, use_cookies, verify, user_agent:
        Passed through to underlying download() calls.
    skip_download:
        If True, do not download; just return a list of GoogleDriveFileToDownload
        (with local_path fields filled) for what *would* be downloaded.
    resume:
        If True, skip files whose local_path already exists.
    manifest_path:
        If set, write a JSON manifest of downloaded files and use it to assist resume.
    workers:
        None or 1  -> sequential download
        "auto"     -> ThreadPoolExecutor with sensible max_workers
        int > 1    -> ThreadPoolExecutor with that many workers

    Returns
    -------
    - If skip_download is True: List[GoogleDriveFileToDownload]
    - Else:
        - If exactly one file downloaded: str (its local path)
        - If multiple files downloaded: List[str]
        - If nothing downloaded: None
    """
    if output is None:
        output = os.getcwd()

    if id is None and url is None:
        raise ValueError("Either url or id must be specified.")

    session = _get_session(proxy=proxy, use_cookies=use_cookies, verify=verify, user_agent=user_agent)

    root = _get_directory_structure(
        url=url,
        id=id,
        session=session,
        use_cookies=use_cookies,
        verify=verify,
        user_agent=user_agent,
    )

    files = _walk_drive_tree(root)

    if len(files) > MAX_NUMBER_FILES:
        raise FolderContentsMaximumLimitError(
            "Folder contains too many files ({}). "
            "Use MAX_NUMBER_FILES to adjust the limit.".format(len(files))
        )

    # Fill local_path and apply .git filter
    for i, f in enumerate(files):
        # logical path inside folder
        path_inside = f.path
        if _is_git_path(path_inside):
            continue

        local_path = osp.join(output, path_inside)
        files[i] = GoogleDriveFileToDownload(id=f.id, path=path_inside, local_path=local_path)

    files = [f for f in files if f.local_path is not None and not _is_git_path(f.path)]

    # Early return if just enumerating
    if skip_download:
        if not quiet:
            print("Skipping download (skip_download=True), {} files listed.".format(len(files)), file=sys.stderr)
        return files

    # Load manifest if any
    manifest: Dict[str, Any] = {}
    if manifest_path is not None:
        manifest = _load_manifest(manifest_path)

    # Build tasks
    tasks: List[Dict[str, Any]] = []
    for f in files:
        if resume and osp.exists(f.local_path):
            if not quiet:
                print("Skipping existing file (resume=True): {}".format(f.local_path), file=sys.stderr)
            continue

        # Ensure parent directory exists
        os.makedirs(osp.dirname(f.local_path), exist_ok=True)

        tasks.append(
            dict(
                id=f.id,
                url=_id_to_download_url(f.id),
                local_path=f.local_path,
                path=f.path,
            )
        )

    if not tasks:
        if not quiet:
            print("No files to download.", file=sys.stderr)
        return None

    # Decide worker count
    max_workers: Optional[int]
    if workers in (None, 1):
        max_workers = None
    elif workers == "auto":
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu * 2)
    elif isinstance(workers, int) and workers > 1:
        max_workers = workers
    else:
        raise ValueError("workers must be None, 1, 'auto', or an int > 1")

    def _download_one(task: Dict[str, Any]) -> Dict[str, Any]:
        file_id = task["id"]
        url = task["url"]
        local_path = task["local_path"]
        path_inside = task["path"]

        attempt = 0
        while True:
            attempt += 1
            try:
                if not quiet:
                    print("Downloading {} -> {}".format(path_inside, local_path), file=sys.stderr)
                # We call gdown.download, which handles cookies/confirm tokens.
                download(
                    url=url,
                    output=local_path,
                    quiet=quiet,
                    proxy=proxy,
                    speed=speed,
                    use_cookies=use_cookies,
                    verify=verify,
                    user_agent=user_agent,
                    session=session,
                )
                status = "ok"
                break
            except Exception as e:
                if attempt >= FILE_RETRY_COUNT:
                    status = "error: {}".format(e)
                    if not quiet:
                        print(
                            "Failed to download {} after {} attempts: {}".format(
                                path_inside, attempt, e
                            ),
                            file=sys.stderr,
                        )
                    break
                if not quiet:
                    print(
                        "Error downloading {} (attempt {}/{}): {}. Retrying in {}s...".format(
                            path_inside, attempt, FILE_RETRY_COUNT, e, FILE_RETRY_SLEEP
                        ),
                        file=sys.stderr,
                    )
                time.sleep(FILE_RETRY_SLEEP)

        return dict(
            id=file_id,
            path=path_inside,
            local_path=local_path,
            status=status,
        )

    results: List[Dict[str, Any]] = []

    if max_workers is None:
        # Sequential
        for t in tasks:
            results.append(_download_one(t))
    else:
        if not quiet:
            print(
                "Downloading {} files with {} workers".format(len(tasks), max_workers),
                file=sys.stderr,
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_download_one, t): t for t in tasks}
            for future in as_completed(future_to_task):
                res = future.result()
                results.append(res)

    # Update manifest
    if manifest_path is not None:
        manifest.setdefault("files", [])
        # Build a dict keyed by (id, path) for easy updates
        index: Dict[str, Dict[str, Any]] = {}
        for entry in manifest["files"]:
            key = "{}::{}".format(entry.get("id"), entry.get("path"))
            index[key] = entry

        for res in results:
            key = "{}::{}".format(res["id"], res["path"])
            index[key] = dict(
                id=res["id"],
                path=res["path"],
                local_path=res["local_path"],
                status=res["status"],
            )

        manifest["files"] = list(index.values())
        _save_manifest(manifest_path, manifest)

    # Collect successful local paths
    local_paths = [r["local_path"] for r in results if r.get("status") == "ok"]

    if not local_paths:
        return None
    if len(local_paths) == 1:
        return local_paths[0]
    return local_paths