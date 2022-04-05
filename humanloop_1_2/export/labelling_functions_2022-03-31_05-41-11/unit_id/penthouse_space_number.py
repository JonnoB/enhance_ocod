import re

def penthouse_space_number(row: Datapoint) -> List[Span]:
    """ 
    Mark the first match of a regex pattern within
    the document text as a span 
    """
    search_pattern = re.compile(r"(?<=penthouse\s)\d+")
    
    match = re.search(search_pattern, row.text)
    
    if match:
        return Span(start=match.start(), end=match.end())
