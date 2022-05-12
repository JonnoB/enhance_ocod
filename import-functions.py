#!/usr/bin/env python3
"""
Humanloop Programmatic function re-import script

Usage: 

``` 
./import-functions.py --project <path to project directory> --functions <path to function export directory> [--label label_name]
``` 

**NB: Corresponding labels must already exist in the project functions are being imported into.**

`--project` is the directory containing a humanloop project, identical to the argument passed into `humanloop run`.

`--functions` can be passed either:
(1) the top-level folder containing subfolders with labelling functions in,
stored by name of label, _or_ 
(2) a single subdirectory of such a folder containing labelling functions for a single label.

`--label`: in the case where a folder of labelling functions for the same single label is supplied (case 2 above),
this flag can be used to override which label the functions are uploaded for. For example, this invocation:

```
./import-functions.py --project my_project --functions my_exported_functions/label_one --label label_two
./import-functions.py --project ./humanloop_1_3 --functions ./humanloop_1_2/export/labelling_functions_2022-04-29_16-29-07
./import-functions.py --project ./humanloop_2 --functions ./humanloop_1_test/export/labelling_functions_2022-05-10_06-40-53
```

would upload functions exported for label_one under the folder label_two. 

Function names will be uniquely suffixed where they clash with existing functions.
"""
import argparse
from pathlib import Path
import sqlite3
import sys
import random
import string


def random_letters():
    return "".join(random.choice(string.ascii_lowercase) for i in range(8))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Import exported functions into a programmatic project"
    )
    parser.add_argument("--project", help="Path to project", required=True)
    parser.add_argument(
        "--functions", help="Path to exported functions directory", required=True
    )
    parser.add_argument(
        "--label",
        help="If provided, import functions under this label name rather than their folder name",
    )

    return parser


def get_database_connection(project):
    """
    Connect to a programmatic project's database
    """
    project_db_file = Path(project) / ".humanloop" / "data.db"

    if not project_db_file.exists():
        print(f"Error - no database file found at {project_db_file}")
        sys.exit(1)

    return sqlite3.connect(project_db_file)


def get_per_label_functions(functions):
    """
    Make a dictionary of label name to function files
    """
    subdirectories = [entry for entry in functions.iterdir() if entry.is_dir()]

    if not subdirectories:
        return {functions.name: list(functions.glob("*.py"))}

    else:
        return {
            subdirectory.name: list(subdirectory.glob("*.py"))
            for subdirectory in subdirectories
        }


def get_id_for_label(connection, label_name):
    """
    Fetch the ID of the given label
    """
    result = connection.execute(
        "SELECT id FROM labels WHERE name = ?", (label_name,)
    ).fetchone()

    if result is None:
        print(f"Error: - Label {label_name} does not exist")
        raise ValueError
    else:
        return result[0]


def make_unique(connection, name):
    """
    Check if a LF name already exists - if so give it a random suffix
    """
    result = connection.execute(
        "SELECT name FROM labelling_functions WHERE name = ?", (name,)
    ).fetchone()

    if result is None:
        return name
    else:
        return f"{name}_{random_letters()}"


def insert_labelling_function(connection, label_id, name, body):
    """
    Insert a labelling function into the DB
    """
    name = make_unique(connection, name)

    result = connection.execute(
        "INSERT INTO labelling_functions (label_id, name, body, is_valid, disabled, run_status) VALUES(?, ?, ?, ?, ?, ?)",
        (label_id, name, body, True, False, "pending"),
    ).fetchone()


def load_function(function_path):
    """
    Read the contents of a function file and its name
    """
    with function_path.open() as function_file:
        return function_path.stem, function_file.read()


def import_functions(project=None, functions=None, label=None):
    """
    Import functions from either

    - A folder with label-named subfolders containing function files, or
    - A label-named folder containing function files
    """
    functions_by_label_name = get_per_label_functions(Path(functions))

    # In the case where a single directory of functions was passed in,
    # allow for them being uploaded under a *different* label name
    if len(functions_by_label_name) == 1 and label is not None:
        existing_label = next(k for k in functions_by_label_name.keys())
        functions_by_label_name[label] = functions_by_label_name.pop(existing_label)

    connection = get_database_connection(project)

    for label_name, function_paths in functions_by_label_name.items():
        try:
            label_id = get_id_for_label(connection, label_name)
        except ValueError:
            print(f"Skipping label {label_name}")
            continue

        for function_path in function_paths:
            name, body = load_function(function_path)

            insert_labelling_function(connection, label_id, name, body)

    # Commit all changes
    connection.commit()
    connection.close()


if __name__ == "__main__":
    parser = build_parser()

    args = parser.parse_args()

    import_functions(**vars(args))

