import re
from hlprogrammatic_helpers import road_regex

def comma_space_number_words_roadtitle_fn(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """
    road_regex2  = r"(\s([a-z])+('s)?)+(\s"+road_regex+r")?"

    search_pattern = re.compile(r"(?<=,\s)(([0-9]+)([a-z])?)(?="+road_regex2+r")")
    
    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
