import re
from hlprogrammatic_helpers import other_classes_regex

def carpark_id_fn(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    ^(the )?(garage(s)?( space)|parking(\s)?space|parking space(s)?|car park(ing)?( space))\s([a-z0-9\-\.]+)
    ^(the )?(garage|parking|parking space|car park)(s)?(ing?)\s([a-z0-9\-\.]+)
    """
    search_pattern = re.compile(r"^(the )?(garage(s)?( space)|parking(\s)?space|parking space(s)?|car park(ing)?( space))\s([a-z0-9\-\.]+)", flags=re.IGNORECASE)

    coords = [(m.span(9)[0], m.span(9)[1]) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
