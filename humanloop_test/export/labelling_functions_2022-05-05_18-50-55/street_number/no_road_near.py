import re 

def no_road_near(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """
    if row.flat_tag ==False:

      search_pattern = re.compile(r"(at|on|in|adjoining|of|off|and|to|being|,)\s(\d+[a-z]?)\b(?=,| and|;|:|\s\()", flags=re.IGNORECASE)
    
      coords = [(m.span(2)[0], m.span(2)[1]) for m in re.finditer(search_pattern, row.text)]

    else:

       coords = []

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]