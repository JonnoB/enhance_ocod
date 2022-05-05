import re

def park_roads(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile(r"\b[\sa-z']*park"+r"(?=,| and| \(|$)", flags=re.IGNORECASE)

    if (row.commercial_park_tag==False):
    
        coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]
    else:
        coords=[]
    return [Span(start=start, end=end) for start, end in coords]
