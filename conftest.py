"""Root conftest: lets `pytest` run every chapter's tests in one session.

Each chapter folder (chNN/) is a self-contained lab whose modules use plain
names such as `lab`, `standin`, `fullmix` or `questions`. Several chapters
share those names, so in one interpreter the first `import lab` would win for
every later chapter. This file keeps a separate set of chapter-local modules
per chapter: before a chapter's test file is imported, and before each of its
tests runs, the chapter's own folder goes to the front of sys.path and only
that chapter's previously imported local modules are visible in sys.modules.

Running a single chapter (`pytest ch17`) behaves exactly as before.
"""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
_CH = re.compile(r"^(ch\d{2})$")
_saved = {}          # chapter -> {module name: module}
_current = [None]

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _chapter_of(path):
    path = os.path.abspath(str(path))
    rel = os.path.relpath(path, ROOT)
    head = rel.split(os.sep, 1)[0]
    return head if _CH.match(head) else None


def _local_modules():
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and _chapter_of(f):
            out[name] = mod
    return out


def _switch(chapter):
    if chapter == _current[0]:
        return
    if _current[0] is not None:
        _saved[_current[0]] = _local_modules()
    for name in _local_modules():
        del sys.modules[name]
    if chapter is not None:
        sys.modules.update(_saved.get(chapter, {}))
        d = os.path.join(ROOT, chapter)
        while d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    _current[0] = chapter


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        ch = _chapter_of(collector.path)
        if ch:
            _switch(ch)
    yield


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    ch = _chapter_of(item.path)
    if ch:
        _switch(ch)
