import re


def mill(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    "(?<=[0-9],)[a-z'\s]+mill(?=,| and|$|;|:)"
    """
    simple = r"(?<=[0-9])[a-z'\s]+mill"
    comma = r"(?<=[0-9],)[a-z'\s]+mill"
    end = r"(?=,| and|$|;|:)"

    combo = r"(" +simple +r"|"+comma+ r")"+end

    search_pattern = re.compile(combo, flags=re.IGNORECASE)

    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
