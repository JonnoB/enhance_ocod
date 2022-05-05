import re
from hlprogrammatic_helpers import *

def comma_space_multi_words_roadtitle_fn(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """

    
    search_pattern = re.compile(r"(?<=,\s)([a-z]+\s){2,5}("+road_regex+")")
    
    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
