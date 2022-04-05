import re

def is_carpark(row: Datapoint) -> List[Span]:
    """ 
    Mark the first match of a regex pattern within
    the document text as a span 
    """
    search_pattern = re.compile(r"^((garage)|(parking(\s)?space)|(parking space)|(car park(ing)?(\sspace)))")
    
    match = re.search(search_pattern, row.text)
    
    if match:
        return Span(start=match.start(), end=match.end())
