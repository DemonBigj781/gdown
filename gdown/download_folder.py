# ============================================================
# gdown.download_folder
#
# Rebuilt fork-friendly version with:
#   - ThreadPoolExecutor for parallel downloads (workers)
#   - .git folders skipped at scan time and before download
#   - Backwards-compatible _get_session() wrapper
#   - download() called WITHOUT 'session=' kwarg (old-gdown safe)
#   - Simple retry logic (20 attempts, 60s between)
# ============================================================

import collections
import itertools
import json
import os
import os.path as osp
import re
import sys
import time
from typing import Any, Dict, List, Optional, Union

from concurrent.futures import ThreadPoolExecutor, as_completed

import bs4

from .download import _get_session, download
from .exceptions import FolderContentsMaximumLimitError
from .parse_url import is_google_drive_url

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MAX_NUMBER_FILES = 1_000_000          # max files allowed in a folder
FILE_RETRY_COUNT = 20                 # how many times to retry each file
FILE_RETRY_SLEEP = 60                 # seconds between file retries
SCAN_SLEEP = 2                        # seconds to sleep before each subfolder scan

GoogleDriveFileToDownload = collections.namedtuple(
    "GoogleDriveFileToDownload", ("id", "path", "local_path")
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _is_git_path(path: str) -> bool:
    """Return True if the path contains a `.git` directory."""
    parts = path.replace("\\", "/").split("/")
    return any(part == ".git" for part in parts)


class _GoogleDriveFile(object):
    TYPE_FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, id, name, type, children=None):
        self.id = id
        self.name = name
        self.type = type
        self.children = children if children is not None else []

    def is_folder(self):
        return self.type == self.TYPE_FOLDER


def _parse_google_drive_file(url: str, content: str):
    """Extracts information about the current page file and its children."""
    folder_soup = bs4.BeautifulSoup(content, features="html.parser")

    # finds the script tag with window['_DRIVE_ivd'] encoded_data
    encoded_data = None
    for script in folder_soup.select("script"):
        inner_html = script.decode_contents()
        if "_DRIVE_ivd" in inner_html:
            # first js string is _DRIVE_ivd, the second one is the encoded arr
            regex_iter = re.compile(r"'((?:[^'\\]|\\.)*)'").finditer(inner_html)
            # get the second elem in the iter
            try:
                encoded_data = next(itertools.islice(regex_iter, 1, None)).group(1)
            except StopIteration:
                raise RuntimeError("Couldn't find the folder encoded JS string")
            break

    if encoded_data is None:
        raise RuntimeError(
            "Cannot retrieve the folder information from the link. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses. "
        )

    decoded = encoded_data.encode("utf-8").decode("unicode_escape")
    folder_arr = json.loads(decoded)

    folder_contents = [] if folder_arr[0] is None else folder_arr[0]

    sep = " - "
    splitted = folder_soup.title.contents[0].split(sep)
    if len(splitted) >= 2:
        name = sep.join(splitted[:-1])
    else:
        raise RuntimeError(
            "file/folder name cannot be extracted from: {}".format(
                folder_soup.title.contents[0]
            )
        )

    gdrive_file = _GoogleDriveFile(
        id=url.split("/")[-1],
        name=name,
        type=_GoogleDriveFile.TYPE_FOLDER,
    )

    id_name_type_iter = [
        (e[0], e[2].encode("raw_unicode_escape").decode("utf-8"), e[3])
        for e in folder_contents
    ]

    return gdrive_file, id_name_type_iter


def _download_and_parse_google_drive_link(
    sess,
    url: str,
    quiet: bool = False,
    remaining_ok: bool = False,
    verify: Union[bool, str] = True,
):
    """Get folder structure of Google Drive folder URL."""
    if not is_google_drive_url(url):
        raise ValueError("URL must be a Google Drive link")

    # canonicalize the language into English
    if "?" in url:
        url_req = url + "&hl=en"
    else:
        url_req = url + "?hl=en"

    res = sess.get(url_req, verify=verify)
    if res.status_code != 200:
        return False, None

    gdrive_file, id_name_type_iter = _parse_google_drive_file(
        url=url_req, content=res.text
    )

    for child_id, child_name, child_type in id_name_type_iter:
        # ---- hard skip .git folder so we never scan inside it ----
        if child_name == ".git":
            if not quiet:
                print("Skipping .git folder", child_name, file=sys.stderr)
            continue

        if child_type != _GoogleDriveFile.TYPE_FOLDER:
            gdrive_file.children.append(
                _GoogleDriveFile(
                    id=child_id,
                    name=child_name,
                    type=child_type,
                )
            )
            continue

        # Subfolder: sleep a bit, then recurse
        child_url = "https://drive.google.com/drive/folders/" + child_id
        if not quiet:
            print("Retrieving folder", child_id, child_name, file=sys.stderr)

        time.sleep(SCAN_SLEEP)

        ok, child = _download_and_parse_google_drive_link(
            sess=sess,
            url=child_url,
            quiet=quiet,
            remaining_ok=remaining_ok,
            verify=verify,
        )

        if not ok:
            return ok, None

        gdrive_file.children.append(child)

    has_at_least_max_files = len(gdrive_file.children) == MAX_NUMBER_FILES
    if not remaining_ok and has_at_least_max_files:
        message = " ".join(
            [
                "The gdrive folder with url: {url}".format(url=url),
                "has more than {max} files,".format(max=MAX_NUMBER_FILES),
                "gdown can't download more than this limit.",
            ]
        )
        raise FolderContentsMaximumLimitError(message)

    return True, gdrive_file


def _get_directory_structure(gdrive_file: _GoogleDriveFile, previous_path: str):
    """Converts a Google Drive folder structure into a local directory list."""
    directory_structure = []
    for file in gdrive_file.children:
        # avoid path separator inside file names
        file.name = file.name.replace(osp.sep, "_")
        if file.is_folder():
            directory_structure.append((None, osp.join(previous_path, file.name)))
            for i in _get_directory_structure(
                file, osp.join(previous_path, file.name)
            ):
                directory_structure.append(i)
        elif not file.children:
            directory_structure.append((file.id, osp.join(previous_path, file.name)))
    return directory_structure


# ----------------------------------------------------------------------
# Backwards-compatible _get_session wrapper
# ----------------------------------------------------------------------

def _make_session(
    proxy,
    use_cookies,
    user_agent,
    verify,
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


# ----------------------------------------------------------------------
# Main download_folder
# ----------------------------------------------------------------------

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
    manifest_path: Optional[str] = None,  # unused, kept for API compatibility
    workers: Union[None, int, str] = None,
):
    """Downloads entire folder from URL or ID."""
    # XOR: exactly one of url or id must be provided
    if not (id is None) ^ (url is None):
        raise ValueError("Either url or id has to be specified")

    if id is not None:
        url = "https://drive.google.com/drive/folders/{id}".format(id=id)

    if user_agent is None:
        user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        )

    sess = _make_session(
        proxy=proxy,
        use_cookies=use_cookies,
        user_agent=user_agent,
        verify=verify,
    )

    if not quiet:
        print("Retrieving folder contents", file=sys.stderr)

    is_success, gdrive_file = _download_and_parse_google_drive_link(
        sess,
        url,
        quiet=quiet,
        remaining_ok=remaining_ok,
        verify=verify,
    )

    if not is_success:
        print("Failed to retrieve folder contents", file=sys.stderr)
        return None

    if not quiet:
        print("Retrieving folder contents completed", file=sys.stderr)
        print("Building directory structure", file=sys.stderr)

    directory_structure = _get_directory_structure(gdrive_file, previous_path="")

    if not quiet:
        print("Building directory structure completed", file=sys.stderr)

    # Skip any .git paths (safety net on top of scan-level skip)
    directory_structure = [
        (fid, path)
        for (fid, path) in directory_structure
        if not _is_git_path(path)
    ]

    if output is None:
        output = os.getcwd() + osp.sep

    if output.endswith(osp.sep):
        root_dir = osp.join(output, gdrive_file.name)
    else:
        root_dir = output

    # If only listing, just return file descriptions
    if skip_download:
        files_to_return: List[GoogleDriveFileToDownload] = []
        for fid, path in directory_structure:
            if fid is None:
                continue
            local_path = osp.join(root_dir, path)
            files_to_return.append(
                GoogleDriveFileToDownload(id=fid, path=path, local_path=local_path)
            )
        if not quiet:
            print(
                "Skipping download (skip_download=True), {} files listed.".format(
                    len(files_to_return)
                ),
                file=sys.stderr,
            )
        return files_to_return

    # Actually download
    if not osp.exists(root_dir):
        os.makedirs(root_dir)

    # Build tasks
    tasks: List[Dict[str, Any]] = []
    existing_paths: List[str] = []

    for fid, path in directory_structure:
        local_path = osp.join(root_dir, path)

        if fid is None:
            if not osp.exists(local_path):
                os.makedirs(local_path)
            continue

        # resume skip
        if resume and osp.isfile(local_path):
            if not quiet:
                print(
                    "Skipping already downloaded file {}".format(local_path),
                    file=sys.stderr,
                )
            existing_paths.append(local_path)
            continue

        parent_dir = osp.dirname(local_path)
        if parent_dir and not osp.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        tasks.append(
            dict(
                id=fid,
                path=path,
                local_path=local_path,
                url="https://drive.google.com/uc?id=" + fid,
            )
        )

    if not tasks and not existing_paths:
        if not quiet:
            print("No files to download.", file=sys.stderr)
        return None

    # Decide workers
    if workers in (None, 1):
        max_workers = None
    elif workers == "auto":
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu * 2)
    elif isinstance(workers, int) and workers > 1:
        max_workers = workers
    else:
        raise ValueError("workers must be None, 1, 'auto', or an int > 1")

    # ------------------------------------------------------------------
    # Download worker (NO 'session=' kwarg to download(), for compatibility)
    # ------------------------------------------------------------------
    def _download_one(task: Dict[str, Any]) -> Dict[str, Any]:
        fid = task["id"]
        path_inside = task["path"]
        local_path = task["local_path"]
        url_dl = task["url"]

        attempt = 0
        while True:
            attempt += 1
            try:
                if not quiet:
                    print(
                        "Downloading {} -> {}".format(path_inside, local_path),
                        file=sys.stderr,
                    )

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
                )

                if res is None:
                    raise RuntimeError("Download ended unsuccessfully (None returned)")

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
                        "Error downloading {} (attempt {}/{}): {}. "
                        "Retrying in {}s...".format(
                            path_inside,
                            attempt,
                            FILE_RETRY_COUNT,
                            e,
                            FILE_RETRY_SLEEP,
                        ),
                        file=sys.stderr,
                    )
                time.sleep(FILE_RETRY_SLEEP)

        return dict(
            id=fid,
            path=path_inside,
            local_path=local_path,
            status=status,
        )

    results: List[Dict[str, Any]] = []

    if max_workers is None:
        for t in tasks:
            results.append(_download_one(t))
    else:
        if not quiet:
            print(
                "Downloading {} files with {} workers".format(
                    len(tasks), max_workers
                ),
                file=sys.stderr,
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_download_one, t): t for t in tasks}
            for future in as_completed(future_to_task):
                results.append(future.result())

    # Collect local paths
    local_paths = list(existing_paths)
    local_paths.extend(
        [r["local_path"] for r in results if r.get("status") == "ok"]
    )

    if not local_paths:
        return None
    if len(local_paths) == 1:
        return local_paths[0]
    return local_paths