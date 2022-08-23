import re
from hlprogrammatic_helpers import other_classes_regex

def is_other_class(row: Datapoint) -> List[Span]:
    """ 
    Mark the first match of a regex pattern within
    the document text as a span 
    """
    search_pattern = re.compile(r"^(the )?(\bland|"+other_classes_regex+r")[a-z]*\b")
    
    match = re.search(search_pattern, row.text)
    
    if match:
        return Span(start=match.start(), end=match.end())
