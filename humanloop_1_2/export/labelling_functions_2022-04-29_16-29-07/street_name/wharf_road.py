import re
from hlprogrammatic_helpers import road_regex


def wharf_road(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile(r"([a-z']+\s)*\b(wharf|quay(s)?|approach|parade|field(s)?|croft|priory|maltings|locks)(?:,)(?![0-9a-z'\s]+"+road_regex+")", flags=re.IGNORECASE)

    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
