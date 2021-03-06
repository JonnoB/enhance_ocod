import re
from hlprogrammatic_helpers import road_regex


def wharf_building(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile("(([a-z']+\s)+(wharf|quay|croft|priory|maltings|locks|gardens))(?:,)(?=[0-9a-z'\s]+"+road_regex+")", flags=re.IGNORECASE)

    coords = [(m.span(1)[0], m.span(1)[1]) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
