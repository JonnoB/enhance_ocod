import re
from hlprogrammatic_helpers import building_regex

def simple_house(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """

    #How do I add in the adjoining test?
    #This is important as land often says adjoiing
    #but negative look behind must be fixed width
    search_pattern = re.compile(r"(?<!(at|of|on)\s)(\b[a-z']+\s)+"+building_regex+r"(?=,)", flags=re.IGNORECASE)
    
    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
