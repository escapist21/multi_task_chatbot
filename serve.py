from __future__ import annotations

import os

from ui.components import build_app


def str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "7860"))
    except Exception:
        port = 7860

    share = str_to_bool(os.getenv("GRADIO_SHARE"), default=False)
    debug = str_to_bool(os.getenv("DEBUG"), default=False)

    demo = build_app()
    demo.launch(server_name=server_name, server_port=port, share=share, debug=debug)


if __name__ == "__main__":
    main()


