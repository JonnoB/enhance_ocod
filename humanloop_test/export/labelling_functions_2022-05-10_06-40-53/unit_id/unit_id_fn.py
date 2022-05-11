import re
from hlprogrammatic_helpers import other_classes_regex

def unit_id_fn(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile(r"(the )?"+other_classes_regex+r"(s)?\s(\d[a-z0-9\-\.]*)", flags=re.IGNORECASE)

    coords = [(m.span(4)[0], m.span(4)[1]) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
