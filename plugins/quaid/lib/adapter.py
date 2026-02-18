"""Platform adapter layer — decouples Quaid core from any specific host.

Provides an abstract interface that Quaid modules call for:
- Path resolution (home dir, data dir, config dir, etc.)
- Notifications (send messages to the user)
- Credentials (API key lookup)
- Session access (conversation transcripts)
- Platform-specific filtering (HEARTBEAT, gateway messages)

Two concrete adapters ship out of the box:
- StandaloneAdapter: default, works anywhere (~/quaid/)
- OpenClawAdapter: auto-detected when running inside OpenClaw (${QUAID_WORKSPACE}/)

Adapter selection (get_adapter()):
1. QUAID_ADAPTER=openclaw|standalone  (explicit override)
2. CLAWDBOT_WORKSPACE env var set     → OpenClawAdapter
3. `clawdbot` binary on PATH          → OpenClawAdapter
4. default                            → StandaloneAdapter

Tests use set_adapter() / reset_adapter() for isolation.
"""

import abc
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ChannelInfo:
    """User's last active channel information."""
    channel: str      # telegram, whatsapp, discord, etc.
    target: str       # chat id, phone number, channel id
    account_id: str   # account identifier (usually "default")
    session_key: str  # session key for reference


class QuaidAdapter(abc.ABC):
    """Abstract interface for platform-specific behavior."""

    # ---- Paths ----

    @abc.abstractmethod
    def quaid_home(self) -> Path:
        """Root directory for all Quaid data (config, data, logs, etc.)."""
        ...

    def data_dir(self) -> Path:
        return self.quaid_home() / "data"

    def config_dir(self) -> Path:
        return self.quaid_home() / "config"

    def logs_dir(self) -> Path:
        return self.quaid_home() / "logs"

    def journal_dir(self) -> Path:
        return self.quaid_home() / "journal"

    def projects_dir(self) -> Path:
        return self.quaid_home() / "projects"

    def core_markdown_dir(self) -> Path:
        return self.quaid_home()

    # ---- Notifications ----

    @abc.abstractmethod
    def notify(self, message: str, channel_override: Optional[str] = None,
               dry_run: bool = False) -> bool:
        """Send a notification message to the user. Returns True on success."""
        ...

    @abc.abstractmethod
    def get_last_channel(self, session_key: str = "") -> Optional[ChannelInfo]:
        """Get the user's last active channel from session state."""
        ...

    # ---- Credentials ----

    @abc.abstractmethod
    def get_api_key(self, env_var_name: str) -> Optional[str]:
        """Retrieve an API key by environment variable name.

        Resolution chain is adapter-specific. Returns None if not found.
        """
        ...

    # ---- Sessions ----

    @abc.abstractmethod
    def get_sessions_dir(self) -> Optional[Path]:
        """Get the directory containing session transcript files."""
        ...

    def get_session_path(self, session_id: str) -> Optional[Path]:
        """Get the path to a specific session's JSONL file."""
        sessions_dir = self.get_sessions_dir()
        if sessions_dir is None:
            return None
        path = sessions_dir / f"{session_id}.jsonl"
        return path if path.exists() else None

    # ---- Platform filtering ----

    @abc.abstractmethod
    def filter_system_messages(self, text: str) -> bool:
        """Return True if this message should be filtered out of transcripts."""
        ...

    # ---- Gateway config (optional) ----

    def get_gateway_config_path(self) -> Optional[Path]:
        """Path to the gateway config file (if applicable)."""
        return None

    # ---- Identity ----

    def get_repo_slug(self) -> str:
        return "steadman-labs/quaid"

    def get_install_url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.get_repo_slug()}/main/install.sh"


class StandaloneAdapter(QuaidAdapter):
    """Default adapter for standalone Quaid installations.

    - Home dir: QUAID_HOME env or ~/quaid/
    - Notifications: stderr
    - Credentials: env var → .env file in quaid home
    - Sessions: quaid_home/sessions/ (if exists)
    - Filtering: no platform messages to filter
    """

    def __init__(self, home: Optional[Path] = None):
        self._home = home

    def quaid_home(self) -> Path:
        if self._home is not None:
            return self._home
        env = os.environ.get("QUAID_HOME", "").strip()
        return Path(env) if env else Path.home() / "quaid"

    def notify(self, message: str, channel_override: Optional[str] = None,
               dry_run: bool = False) -> bool:
        if os.environ.get("QUAID_DISABLE_NOTIFICATIONS"):
            return True
        if dry_run:
            print(f"[notify] (dry-run) {message}", file=sys.stderr)
            return True
        print(f"[quaid] {message}", file=sys.stderr)
        return True

    def get_last_channel(self, session_key: str = "") -> Optional[ChannelInfo]:
        return None

    def get_api_key(self, env_var_name: str) -> Optional[str]:
        # 1. Environment variable
        key = os.environ.get(env_var_name, "").strip()
        if key:
            return key

        # 2. .env file in quaid home
        env_file = self.quaid_home() / ".env"
        if env_file.exists():
            return _read_env_file(env_file, env_var_name)

        return None

    def get_sessions_dir(self) -> Optional[Path]:
        d = self.quaid_home() / "sessions"
        return d if d.is_dir() else None

    def filter_system_messages(self, text: str) -> bool:
        return False


class OpenClawAdapter(QuaidAdapter):
    """Adapter for running inside the OpenClaw gateway.

    - Home dir: CLAWDBOT_WORKSPACE env or ${QUAID_WORKSPACE}/
    - Notifications: clawdbot message send CLI
    - Credentials: env var → workspace .env → macOS Keychain
    - Sessions: ~/.openclaw/sessions/
    - Filtering: HEARTBEAT, GatewayRestart, System: messages
    """

    _MAIN_SESSION_KEY = "agent:main:main"

    def quaid_home(self) -> Path:
        env = os.environ.get("CLAWDBOT_WORKSPACE", "").strip()
        if env:
            return Path(env)
        raise RuntimeError(
            "CLAWDBOT_WORKSPACE environment variable is not set. "
            "Set it to your OpenClaw workspace path, or use QUAID_ADAPTER=standalone."
        )

    def notify(self, message: str, channel_override: Optional[str] = None,
               dry_run: bool = False) -> bool:
        if os.environ.get("QUAID_DISABLE_NOTIFICATIONS"):
            return True

        info = self.get_last_channel()
        if not info:
            print("[notify] No last channel found", file=sys.stderr)
            return False

        effective_channel = channel_override or info.channel

        cmd = [
            "clawdbot", "message", "send",
            "--channel", effective_channel,
            "--target", info.target,
            "--message", message,
        ]

        if info.account_id and info.account_id != "default":
            cmd.extend(["--account", info.account_id])

        if dry_run:
            print(f"[notify] Would run: {' '.join(cmd)}")
            return True

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"[notify] Sent to {effective_channel}:{info.target}")
                return True
            else:
                print(f"[notify] Send failed: {result.stderr}", file=sys.stderr)
                return False
        except subprocess.TimeoutExpired:
            print("[notify] Send timed out", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[notify] Error: {e}", file=sys.stderr)
            return False

    def get_last_channel(self, session_key: str = "") -> Optional[ChannelInfo]:
        session_key = session_key or self._MAIN_SESSION_KEY
        sessions_path = self._find_sessions_json()
        if not sessions_path:
            return None

        try:
            with open(sessions_path) as f:
                sessions = json.load(f)

            session = sessions.get(session_key)
            if not session:
                return None

            channel = session.get("lastChannel")
            target = session.get("lastTo")
            account_id = session.get("lastAccountId", "default")

            if not channel or not target:
                return None

            return ChannelInfo(
                channel=channel,
                target=target,
                account_id=account_id,
                session_key=session_key,
            )
        except (json.JSONDecodeError, IOError) as e:
            print(f"[notify] Error reading sessions: {e}", file=sys.stderr)
            return None

    def get_api_key(self, env_var_name: str) -> Optional[str]:
        # 1. Environment variable
        key = os.environ.get(env_var_name, "").strip()
        if key:
            return key

        # 2. .env file in workspace root
        env_file = self.quaid_home() / ".env"
        if env_file.exists():
            found = _read_env_file(env_file, env_var_name)
            if found:
                return found


        return None

    def get_sessions_dir(self) -> Optional[Path]:
        d = Path.home() / ".openclaw" / "sessions"
        return d if d.is_dir() else None

    def filter_system_messages(self, text: str) -> bool:
        if text.startswith("GatewayRestart:") or text.startswith("System:"):
            return True
        if '"kind": "restart"' in text:
            return True
        if "HEARTBEAT" in text and "HEARTBEAT_OK" in text:
            return True
        if re.sub(r"[*_<>/b\s]", "", text).startswith("HEARTBEAT_OK"):
            return True
        return False

    def get_gateway_config_path(self) -> Optional[Path]:
        p = Path.home() / ".openclaw" / "openclaw.json"
        return p if p.exists() else None

    # ---- Internal helpers ----

    def _find_sessions_json(self) -> Optional[Path]:
        """Find the agent sessions.json file."""
        candidates = [
            Path.home() / ".clawdbot" / "agents" / "main" / "sessions" / "sessions.json",
            Path.home() / ".openclaw" / "agents" / "main" / "sessions" / "sessions.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    @staticmethod
    def _keychain_lookup(service: str, account: str) -> Optional[str]:
        """Keychain not available in standalone mode."""
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _read_env_file(env_file: Path, var_name: str) -> Optional[str]:
    """Read a variable from a .env file.

    Handles: KEY=value, KEY="value", KEY='value', inline # comments,
    comment lines, and empty values.  Does NOT handle ``export`` prefix
    or multi-line values (these are uncommon in .env files).
    """
    prefix = f"{var_name}="
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or not line.startswith(prefix):
                continue
            value = line.split("=", 1)[1].strip()
            # Quoted values: extract content between matching quotes
            if value.startswith('"') and '"' in value[1:]:
                value = value[1:value.index('"', 1)]
            elif value.startswith("'") and "'" in value[1:]:
                value = value[1:value.index("'", 1)]
            else:
                # Unquoted: strip inline comments
                if " #" in value:
                    value = value[:value.index(" #")].rstrip()
            if value:
                return value
    except (IOError, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_adapter: Optional[QuaidAdapter] = None


def get_adapter() -> QuaidAdapter:
    """Get the current adapter (auto-detected on first call).

    Selection order:
    1. QUAID_ADAPTER=openclaw|standalone  (explicit)
    2. CLAWDBOT_WORKSPACE env set         → OpenClawAdapter
    3. `clawdbot` on PATH                 → OpenClawAdapter
    4. default                            → StandaloneAdapter
    """
    global _adapter
    if _adapter is not None:
        return _adapter

    explicit = os.environ.get("QUAID_ADAPTER", "").lower()
    if explicit == "openclaw":
        _adapter = OpenClawAdapter()
    elif explicit == "standalone":
        _adapter = StandaloneAdapter()
    elif os.environ.get("CLAWDBOT_WORKSPACE"):
        _adapter = OpenClawAdapter()
    elif shutil.which("clawdbot"):
        _adapter = OpenClawAdapter()
    else:
        _adapter = StandaloneAdapter()

    return _adapter


def set_adapter(adapter: QuaidAdapter) -> None:
    """Override the adapter (for tests)."""
    global _adapter
    _adapter = adapter


def reset_adapter() -> None:
    """Reset to auto-detection (for tests).

    Also clears cached API keys and config so they re-resolve
    against the next adapter.
    """
    global _adapter
    _adapter = None
    # Clear API key cache so it re-resolves against the new adapter
    try:
        from llm_clients import clear_api_key_cache
        clear_api_key_cache()
    except ImportError:
        pass
