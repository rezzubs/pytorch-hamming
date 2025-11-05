import typer

from . import record
from . import draw_mean

from .utils import setup_logging

app = typer.Typer()

app.add_typer(record.app)
app.add_typer(draw_mean.app)


def main():
    setup_logging()
    app()


if __name__ == "__main__":
    main()
