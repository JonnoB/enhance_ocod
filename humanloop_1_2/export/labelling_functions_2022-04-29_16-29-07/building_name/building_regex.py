import re


def building_regex(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile(r"(?<!\S)(?:(?!\b(?:at|of|and)\b)[^\n\d\),])*? building(s)?\b(?=,| and|$|;|:)", flags=re.IGNORECASE)

    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
