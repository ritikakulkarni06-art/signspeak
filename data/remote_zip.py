"""Read single files out of a zip archive without downloading the whole archive.

Both datasets ship as large zips (INCLUDE: ~1.5 GB per category part; ASL
Citizen: one 46 GB file) but only a few hundred clips are needed. A zip keeps
its directory at the end of the file, so with HTTP range requests we read that
directory once and then fetch exactly the bytes of each clip. A local zip file
goes through the same code.
"""

from __future__ import annotations

import http.client
import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

_USER_AGENT = "signspeak-data-prep"
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}
_READ_AHEAD = 1 << 20  # bytes fetched per request while parsing the zip directory
_LOCAL_HEADER = struct.Struct("<4s5H3I2H")  # fixed 30-byte part of a zip local file header
_LOCAL_EXTRA_SLACK = 256  # guess at the local header's extra field, so most reads take one request


class HttpSource:
    """Byte ranges of a file on a web server that supports range requests."""

    def __init__(self, url: str, retries: int = 8, timeout: float = 120):
        self.url = url
        self._retries = retries
        self._timeout = timeout
        self.size = self._fetch(0, 1, want_total_size=True)

    def read_range(self, start: int, length: int) -> bytes:
        return self._fetch(start, length)

    def _fetch(self, start: int, length: int, want_total_size: bool = False):
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": _USER_AGENT}
        )
        for attempt in range(self._retries):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    if response.status != 206:
                        raise OSError(f"{self.url} ignored the range request (HTTP {response.status})")
                    if want_total_size:
                        return int(response.headers["Content-Range"].rsplit("/", 1)[1])
                    data = response.read()
                if len(data) != length:
                    raise ConnectionError(f"short read: got {len(data)} of {length} bytes")
                return data
            except urllib.error.HTTPError as error:
                if error.code not in _RETRYABLE_HTTP or attempt == self._retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
                if attempt == self._retries - 1:
                    raise
            time.sleep(min(2**attempt, 60))


class LocalSource:
    """Byte ranges of a local file. Opens the file per read, so it is thread-safe
    and never holds a handle open."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size

    def read_range(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as file:
            file.seek(start)
            data = file.read(length)
        if len(data) != length:
            raise OSError(f"short read from {self.path}")
        return data


class _SourceFile(io.RawIOBase):
    """Seekable file view over a source, so zipfile can parse the directory."""

    def __init__(self, source):
        self._source = source
        self._pos = 0
        self._block_start = 0
        self._block = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._source.size}[whence]
        self._pos = base + offset
        return self._pos

    def readinto(self, buffer) -> int:
        want = min(len(buffer), self._source.size - self._pos)
        if want <= 0:
            return 0
        offset = self._pos - self._block_start
        if offset < 0 or offset + want > len(self._block):
            self._block_start = self._pos
            self._block = self._source.read_range(self._pos, min(max(want, _READ_AHEAD), self._source.size - self._pos))
            offset = 0
        buffer[:want] = self._block[offset : offset + want]
        self._pos += want
        return want


class RemoteZip:
    """A zip archive at a URL or local path. Lists its files and reads any one of
    them with a single range request. read() is safe to call from several threads."""

    def __init__(self, location: str | Path):
        location = str(location)
        self.location = location
        self._source = HttpSource(location) if location.startswith(("http://", "https://")) else LocalSource(location)
        with zipfile.ZipFile(_SourceFile(self._source)) as archive:
            self._infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}

    @property
    def names(self) -> list[str]:
        return list(self._infos)

    def read(self, name: str) -> bytes:
        info = self._infos[name]
        if info.flag_bits & 0x1:
            raise ValueError(f"{name} is encrypted")
        # The local header repeats the name and has its own extra field, whose
        # length we only learn once we read it; fetch a little slack with the data.
        span = _LOCAL_HEADER.size + len(info.orig_filename.encode()) + _LOCAL_EXTRA_SLACK + info.compress_size
        blob = self._source.read_range(info.header_offset, min(span, self._source.size - info.header_offset))
        signature, *_, name_length, extra_length = _LOCAL_HEADER.unpack_from(blob)
        if signature != b"PK\x03\x04":
            raise OSError(f"bad local header for {name}")
        start = _LOCAL_HEADER.size + name_length + extra_length
        data = blob[start : start + info.compress_size]
        if len(data) < info.compress_size:
            data += self._source.read_range(info.header_offset + start + len(data), info.compress_size - len(data))

        if info.compress_type == zipfile.ZIP_STORED:
            content = data
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            content = zlib.decompress(data, -15)
        else:
            raise ValueError(f"{name} uses unsupported compression type {info.compress_type}")
        if zlib.crc32(content) != info.CRC:
            raise OSError(f"CRC mismatch for {name}")
        return content
