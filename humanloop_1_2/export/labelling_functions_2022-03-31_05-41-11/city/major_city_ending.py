import re
from hlprogrammatic_helpers import city_regex

def major_city_ending(row: Datapoint) -> List[Span]:
    """ 
    Mark the first match of a regex pattern within
    the document text as a span 
    """
    search_pattern = re.compile(r"(?<=\b)[a-z]+$")
    
    match = re.search(search_pattern, row.text)
    
    if match:
        return Span(start=match.start(), end=match.end())
