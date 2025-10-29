import typer

from . import record

from .utils import setup_logging

app = typer.Typer()

app.add_typer(record.app)


def main():
    setup_logging()
    app()


if __name__ == "__main__":
    main()
