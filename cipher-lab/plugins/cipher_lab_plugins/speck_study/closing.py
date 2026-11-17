"""Two manual-reset Windows events: parent request and saved/quiesced child ACK.

Poll is WaitForSingleObject(timeout=0), never waits for the child. Handles are
owned externally (the supervising parent); this transport cannot certify a
RunStore terminal or substitute for independently verified child exit.
"""
from __future__ import annotations

import ctypes
import os
from typing import Any


class WindowsEvents:
    def __init__(self, library: Any = None) -> None:
        if library is None:
            if os.name != 'nt':
                raise RuntimeError('ENVIRONMENT_BLOCKED')
            from ctypes import wintypes as W

            library = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]
            library.CreateEventW.argtypes = [ctypes.c_void_p, W.BOOL, W.BOOL, W.LPCWSTR]
            library.CreateEventW.restype = W.HANDLE
            library.CloseHandle.argtypes = [W.HANDLE]
            library.CloseHandle.restype = W.BOOL
            library.SetEvent.argtypes = [W.HANDLE]
            library.SetEvent.restype = W.BOOL
            library.WaitForSingleObject.argtypes = [W.HANDLE,W.DWORD]
            library.WaitForSingleObject.restype = W.DWORD
        self.library = library

    def create(self) -> int:
        handle = self.library.CreateEventW(None, True, False, None)
        if not handle:
            raise RuntimeError('CLOSING_TRANSPORT_FAILED')
        try:
            getattr(os, 'set_handle_inheritable')(handle, True)
        except BaseException:
            self.close(handle)
            raise
        return int(handle)

    def close(self, handle: int) -> None:
        if not self.library.CloseHandle(handle):
            raise RuntimeError('CLOSING_TRANSPORT_FAILED')

    def set(self, handle: int) -> None:
        if not self.library.SetEvent(handle):
            raise RuntimeError('CLOSING_TRANSPORT_FAILED')

    def poll(self, handle: int) -> bool:
        result = self.library.WaitForSingleObject(handle,0)
        if result not in (0,258):  # WAIT_OBJECT_0 / WAIT_TIMEOUT only
            raise RuntimeError('CLOSING_TRANSPORT_FAILED')
        return bool(result == 0)


class ClosingChannel:
    def __init__(self, api: Any, request_handle: int, ack_handle: int) -> None:
        if (type(request_handle) is not int or request_handle <= 0
                or type(ack_handle) is not int or ack_handle <= 0
                or ack_handle == request_handle):
            raise ValueError('invalid distinct event handles')
        self.api, self.request_handle, self.ack_handle = api,request_handle,ack_handle

    def request(self) -> None:
        self.api.set(self.request_handle)

    def requested(self) -> bool:
        return bool(self.api.poll(self.request_handle))

    def acknowledged(self) -> bool:
        return bool(self.api.poll(self.ack_handle))

    def acknowledge(self, *, saved: bool, quiesced: bool) -> None:
        if saved is not True or quiesced is not True or not self.requested():
            raise RuntimeError('CLOSING_NOT_SAVED_QUIESCENT')
        self.api.set(self.ack_handle)


class EventPair:
    """Parent owns exactly two inheritable events; Popen has an explicit whitelist."""
    def __init__(self, api: Any = None) -> None:
        self.api = WindowsEvents() if api is None else api
        self.handle_list: list[int] = []
        try:
            self.handle_list.append(self.api.create())
            self.handle_list.append(self.api.create())
        except BaseException:
            self.close()
            raise
        self.channel = ClosingChannel(self.api, *self.handle_list)

    def spawn_options(self) -> dict[str, Any]:
        # Python subprocess official contract: handle_list requires close_fds=True.
        startup = type('Startup', (), {})() if os.name != 'nt' else __import__(
            'subprocess').STARTUPINFO()
        startup.lpAttributeList = {'handle_list': list(self.handle_list)}
        return dict(close_fds=True, startupinfo=startup)

    def close(self) -> None:
        errors = []
        for handle in list(self.handle_list):
            try:
                self.api.close(handle)
                self.handle_list.remove(handle)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError('CLOSING_TRANSPORT_FAILED') from errors[0]
