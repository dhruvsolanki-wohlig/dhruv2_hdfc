"""Language-specific modules for the PDF translation service.

Each language has its own folder (e.g. ``languages/hindi/``) containing a
``LanguageHandler`` subclass that isolates all language-specific behavior:

* Configuration (font family, script, code)
* Glossary (locked insurance terminology)
* Translation prompt customisations
* Rendering overrides (vector/PIL shaping, text wrapping, line breaks)
* Pre/post-processing hooks
* Residual-English detection (language-specific)

The shared base class and dispatcher live in ``languages/base.py``.
Use ``get_language_handler(language_name)`` to obtain the handler for a
given language.

All language handlers are auto-imported on package load so they register
themselves in the dispatcher's registry.
"""
from .base import LanguageHandler, get_language_handler, register_language, _REGISTRY

# ── Auto-import all language modules so they self-register ──────────────────
import importlib as _importlib
import pkgutil as _pkgutil

for _finder, _name, _ispkg in _pkgutil.iter_modules(__path__):
    if _name.startswith("_"):
        continue
    _importlib.import_module(f".{_name}", package=__name__)

__all__ = ["LanguageHandler", "get_language_handler", "register_language"]