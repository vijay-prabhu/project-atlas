"""Load versioned prompt templates from disk.

Templates live at ``data/prompts/v{version}/{name}.txt`` and use
``{variable_name}`` placeholders filled at runtime.
"""

import difflib
from pathlib import Path
from typing import Optional

from src.core.observability import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _PROJECT_ROOT / "data" / "prompts"


class PromptRegistry:
    """Loads, caches, and fills versioned prompt templates.

    Usage::

        registry = PromptRegistry()
        prompt = registry.get_prompt(
            "classify_document",
            version=2,
            document_text="...",
            categories="A, B, C",
        )
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self._prompts_dir = prompts_dir or _PROMPTS_DIR
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prompt(self, name: str, version: int = 1, **kwargs: str) -> str:
        """Load a prompt template and fill its placeholders.

        Args:
            name: Template name (file stem without extension).
            version: Prompt version number.
            **kwargs: Values for ``{placeholder}`` substitution.

        Returns:
            The rendered prompt string.

        Raises:
            FileNotFoundError: If the template file doesn't exist.
            KeyError: If a required placeholder is missing from kwargs.
        """
        template = self._load_template(name, version)
        try:
            rendered = template.format(**kwargs)
        except KeyError as exc:
            raise KeyError(
                f"Missing placeholder {exc} in prompt '{name}' v{version}. "
                f"Provide it as a keyword argument.",
            ) from exc

        logger.info(
            "Prompt rendered",
            extra={"prompt": name, "version": version},
        )
        return rendered

    def list_versions(self, name: str) -> list[int]:
        """Return sorted list of available version numbers for a prompt name."""
        versions: list[int] = []
        if not self._prompts_dir.exists():
            return versions

        for version_dir in self._prompts_dir.iterdir():
            if not version_dir.is_dir():
                continue
            dir_name = version_dir.name
            if not dir_name.startswith("v"):
                continue
            try:
                v = int(dir_name[1:])
            except ValueError:
                continue
            template_path = version_dir / f"{name}.txt"
            if template_path.exists():
                versions.append(v)

        return sorted(versions)

    def compare_versions(self, name: str, v1: int, v2: int) -> dict:
        """Return a unified diff between two prompt versions.

        Returns:
            {
                "name": str,
                "v1": int,
                "v2": int,
                "diff": list[str],   # unified diff lines
                "v1_lines": int,
                "v2_lines": int,
            }
        """
        text_v1 = self._load_template(name, v1)
        text_v2 = self._load_template(name, v2)

        diff = list(
            difflib.unified_diff(
                text_v1.splitlines(keepends=True),
                text_v2.splitlines(keepends=True),
                fromfile=f"v{v1}/{name}.txt",
                tofile=f"v{v2}/{name}.txt",
            ),
        )

        return {
            "name": name,
            "v1": v1,
            "v2": v2,
            "diff": diff,
            "v1_lines": len(text_v1.splitlines()),
            "v2_lines": len(text_v2.splitlines()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_template(self, name: str, version: int) -> str:
        """Load a template from disk, using the in-memory cache."""
        cache_key = f"v{version}/{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._prompts_dir / f"v{version}" / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {path}",
            )

        template = path.read_text(encoding="utf-8")
        self._cache[cache_key] = template
        logger.info(
            "Prompt template loaded",
            extra={"prompt": name, "version": version, "path": str(path)},
        )
        return template
