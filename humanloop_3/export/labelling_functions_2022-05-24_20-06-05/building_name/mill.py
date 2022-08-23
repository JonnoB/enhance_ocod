import re


def mill(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    "(?<=[0-9],)[a-z'\s]+mill(?=,| and|$|;|:)"
    """
    mill_regex = r"(?<=[0-9]|,|\))\s([a-z][a-z'\s]+mill)"
    simple = r"(?<=[0-9])\s([a-z][a-z'\s]+mill)"
    comma = r"(?<=[0-9],)\s([a-z][a-z'\s]+mill)"
    end = r"(?=,| and|$|;|:)"

    combo = r"(" +simple +r"|"+comma+ r")"+end

    search_pattern = re.compile(mill_regex+end, flags=re.IGNORECASE)

    coords = [(m.span(1)[0], m.span(1)[1]) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
