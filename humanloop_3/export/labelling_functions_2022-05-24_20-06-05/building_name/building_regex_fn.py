import re
from hlprogrammatic_helpers import building_regex
from hlprogrammatic_helpers import multi_word_no_land
def building_regex_fn(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """

    #How do I add in the adjoining test?
    #This is important as land often says adjoiing(?=,| \(| and)
    #but negative look behind must be fixed width\b(?=,| and|$|;|:|\()
    search_pattern = re.compile(multi_word_no_land+building_regex+r"\b(?=,| and|$|;|:|\(|\s\d+,)", flags=re.IGNORECASE)
    
    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
