import re
from hlprogrammatic_helpers import multi_word_no_land
from hlprogrammatic_helpers import waygate_regex

def words_waygate_fn(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile(multi_word_no_land+waygate_regex+r"(?=,| and| \()")
    
    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
