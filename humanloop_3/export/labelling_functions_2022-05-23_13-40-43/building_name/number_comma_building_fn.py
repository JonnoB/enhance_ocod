import re
from hlprogrammatic_helpers import starting_building
from hlprogrammatic_helpers import xx_to_yy_regex
def number_comma_building_fn(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    "^(?:flat|apartments|penthouse)(?:s)?\s(?:[a-z]|\d+[a-z0-9\-\.]*|"+xx_to_yy_regex+")(?:,)[a-z\-']+\b(?=,|$|;|:|\()"
    """
    
    #How do I add in the adjoining test?
    #This is important as land often says adjoiing(?=,| \(| and)
    #but negative look behind must be fixed width\b(?=,| and|$|;|:|\()
    search_pattern = re.compile("(?:flat|apartment|unit)s? (?:[a-z]|\d+[a-z0-9\-\.]*|"+xx_to_yy_regex+"),\s([a-z\s.']+)(?=,)", flags=re.IGNORECASE)
    
    coords = [(m.span(1)[0], m.span(1)[1]) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
