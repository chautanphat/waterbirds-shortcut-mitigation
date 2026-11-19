"""Bounded parallel HTTP range stream for checksum-verified public archives."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import requests


class RangeStream:
    def __init__(self, origin, resolved_url, length, *, workers=8, chunk=4*1024*1024):
        self.origin,self.url,self.length=origin,resolved_url,length
        self.chunk,self.offset,self.position=chunk,0,0
        self.buffer=b''
        self.lock=threading.Lock()
        self.url_time=time.monotonic()
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.pending={}
        self.next_start=0
        for _ in range(workers):self._schedule()

    def _schedule(self):
        start=self.next_start
        if start>=self.length:return
        self.next_start+=self.chunk
        self.pending[start]=self.pool.submit(self._fetch,start,min(start+self.chunk,self.length)-1)

    def _fetch(self,start,end):
        for attempt in range(4):
            with self.lock:
                if time.monotonic()-self.url_time>35:
                    # Mint another public signed HTTPS redirect; never disable TLS.
                    r=requests.get(self.origin,stream=True,timeout=120)
                    r.raise_for_status()
                    if not r.url.startswith('https://'):
                        r.close();raise ValueError('Archive must resolve to HTTPS')
                    self.url,self.url_time=r.url,time.monotonic()
                    r.close()
                url=self.url
            try:
                r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=120)
                if r.status_code==403:
                    with self.lock:self.url_time=0
                    continue
                r.raise_for_status()
                expected=f'bytes {start}-{end}/{self.length}'
                if r.status_code!=206 or r.headers.get('Content-Range')!=expected or len(r.content)!=end-start+1:
                    raise ValueError('Server returned an invalid archive range')
                return r.content
            except requests.RequestException:
                if attempt==3:raise
        raise RuntimeError('Could not retrieve authenticated public archive range')

    def read(self,n=-1):
        if n<0:n=self.length-self.position
        parts=[]
        while n and self.position<self.length:
            if not self.buffer:
                self.buffer=self.pending.pop(self.offset).result()
                self.offset+=len(self.buffer)
                self._schedule()
            take=min(n,len(self.buffer))
            parts.append(self.buffer[:take]);self.buffer=self.buffer[take:]
            n-=take;self.position+=take
        return b''.join(parts)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
