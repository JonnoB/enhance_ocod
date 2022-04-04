import re 
from hlprogrammatic_helpers import multi_word_no_land

def tower_match(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile(multi_word_no_land+r" (tower(s)?)(\s\w+)?(?=,| and)", flags=re.IGNORECASE)
    
    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
