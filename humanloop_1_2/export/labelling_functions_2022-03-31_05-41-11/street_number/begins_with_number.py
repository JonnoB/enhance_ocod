import re 

def begins_with_number(row: Datapoint) -> List[Span]:
    """ 
    Mark the first match of a regex pattern within
    the document text as a span 
    """
    #if not a flat and start with a number, that is a street number.
    #otherwise not a street number
    if row.flat_tag == False:
        search_pattern = re.compile(r"^(\d+([a-z])?)(?!(\s?(to|-)\s?)(\d+)\b)")
    
        match = re.search(search_pattern, row.text)
    else:
        match = None

    if match:
        return Span(start=match.start(), end=match.end())