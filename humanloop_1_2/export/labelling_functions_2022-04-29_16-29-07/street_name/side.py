import re


def side(row: Datapoint) -> List[Span]:
    """
    roads which for some reason are called side
    """
    search_pattern = re.compile("(new road|lodge|carr moor|lake|chase|thames|kennet|coppice|church|pool) side", flags=re.IGNORECASE)

    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
