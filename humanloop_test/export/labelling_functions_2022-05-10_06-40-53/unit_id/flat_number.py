import re 

def flat_number(row: Datapoint) -> List[Span]:
    """ 
    matches a flat like property
    followed by a space and number
    followed by a bunch of random stuff
    """
    search_pattern = re.compile("(apartment|flat|penthouse)\s(\d+[a-z0-9\-\.]*)", flags=re.IGNORECASE)
    
    coords = [(m.span(2)[0], m.span(2)[1]) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
