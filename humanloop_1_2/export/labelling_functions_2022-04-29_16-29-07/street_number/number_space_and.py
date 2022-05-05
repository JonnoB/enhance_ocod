import re 

def number_space_and(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """
    if row.flat_tag ==False:

      search_pattern = re.compile(r"(?<=\s|,)(\d+[a-z]?)(?=\sand)", flags=re.IGNORECASE)
    
      coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    else:

       coords = [(m.start(), m.end()) for m in re.finditer("load of old trousers", row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]