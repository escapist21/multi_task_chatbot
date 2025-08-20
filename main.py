from ui.components import build_app


def main() -> None:
    demo = build_app()
    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
