import re 

def and_space_n_flats(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """

    if row.flat_tag ==True:

      search_pattern = re.compile("(?<=and\s)(\d+(\w{1})?)")
    
      coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]
    
    else:

     coords = []

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
