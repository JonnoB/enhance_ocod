import re 

def and_space_n(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """

    if row.flat_tag !=True:

      search_pattern = re.compile(r"(?<=and\s)(\d+)")
    
      coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]
    
    else:

     coords = [(m.start(), m.end()) for m in re.finditer("load_of_old_trousers", row.text)] 

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]