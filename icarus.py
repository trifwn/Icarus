"""
Entry point for the ICARUS CLI.
"""
from cli.cli_home import cli_home
from ICARUS.Database.db import DB


if __name__ == "__main__":
    # Establish DB connection
    db = DB()
    db.load_data()

    cli_home(db)
