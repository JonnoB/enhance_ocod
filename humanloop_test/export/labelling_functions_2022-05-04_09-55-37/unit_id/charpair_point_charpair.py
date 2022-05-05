import re


def charpair_point_charpair(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    \b[a-z0-9]{1,2}\.[a-z0-9]{1,2}\.[a-z0-9]{1,2}\b
    \b[a-z0-9]{1,2}\.((\.)?[a-z0-9]{1,2}){1,}
    """
    search_pattern = re.compile(r"\b[a-z0-9]{1,2}\.[a-z0-9]{1,2}(\.[a-z0-9]{1,2})?\b", flags=re.IGNORECASE)

    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
