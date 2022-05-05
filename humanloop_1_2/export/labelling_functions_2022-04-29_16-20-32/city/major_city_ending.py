import re
from hlprogrammatic_helpers import city_regex

def major_city_ending(row: Datapoint) -> List[Span]:
    """ 
    Mark the first match of a regex pattern within
    the document text as a span 
    at|on|in|adjoining|of|off|to
    r"(?<=\b)[a-z][a-z.\-\s']+[a-z]$"
    r"(?<=\b)[a-z][a-z.\-\s']+[a-z](?! and[a-z\(\)0-9\s])$"
    r"(?<=\b)[a-z][a-z.\-\s][a-z](?:(?!and)[a-z\(\)0-9\s])*$"
    """
    search_pattern = re.compile( r"xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    
    match = re.search(search_pattern, row.text)
    
    if match:
        return Span(start=match.start(), end=match.end())
