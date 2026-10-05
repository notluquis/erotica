"""A read-only, seekable file object over HTTP range requests, with a block cache, so that
``h5py.File(HTTPFile(url))`` reads only the bytes of the datasets that are sliced.

Why: the DECaPS 3D map (Zucker+2025) is an 8.6 GB HDF5 on Harvard Dataverse and Bayestar19 another
multi-GB one; NGC 6383 needs a few hundred of their pixels. Dataverse answers with a 303 to a
pre-signed S3 URL valid for 2 h; the redirect is re-resolved when it expires.
"""

from __future__ import annotations

import io
import time

import requests


class HTTPFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20, max_blocks: int = 512):
        self.url, self.block, self.max_blocks = url, block, max_blocks
        self._resolve()
        self.pos, self.cache, self.n_requests, self.bytes_read = 0, {}, 0, 0

    def _resolve(self):
        r = requests.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
        self.real = r.url
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.t_resolved = time.time()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, k):
        if k in self.cache:
            return self.cache[k]
        if time.time() - self.t_resolved > 5400:
            self._resolve()
        a, b = k * self.block, min(self.size, (k + 1) * self.block) - 1
        for i in range(6):
            try:
                r = requests.get(self.real, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
                if r.status_code in (403, 400):
                    self._resolve()
                    raise OSError(r.status_code)
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    break
                raise OSError(f"HTTP {r.status_code}")
            except (OSError, requests.RequestException):
                if i == 5:
                    raise
                time.sleep(3 * (i + 1))
        self.n_requests += 1
        self.bytes_read += len(r.content)
        if len(self.cache) >= self.max_blocks:
            self.cache.pop(next(iter(self.cache)))
        self.cache[k] = r.content
        return r.content

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            k = p // self.block
            blk = self._block(k)
            s = p - k * self.block
            take = blk[s : s + n - len(out)]
            out += take
            p += len(take)
        b[:n] = out
        self.pos += n
        return n
