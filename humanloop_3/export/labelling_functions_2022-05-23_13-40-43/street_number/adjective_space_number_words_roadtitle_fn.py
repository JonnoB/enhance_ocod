import re
from hlprogrammatic_helpers import road_regex

def adjective_space_number_words_roadtitle_fn(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """
    road_regex2  = r"(\s([a-z])+('s)?)+(\s"+road_regex+r")?"
    search_pattern = re.compile(r"(?:\b(?:at|on|in|adjoining|of|off|and|next to)\s)(\d+[a-z]?)(?="+road_regex2+r")")
    
    coords = [(m.span(1)[0], m.span(1)[1]) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
