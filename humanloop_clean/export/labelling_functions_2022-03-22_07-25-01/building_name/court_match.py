import re 

def court_match(row: Datapoint) -> List[Span]:
    """ 
   Court needs to be treated slightly differently to ensure that places like "earls court road"
   are not acciedntly captured
    """
    search_pattern = re.compile(r"([a-z']+\s)+(court)(?=,|\sand)", flags=re.IGNORECASE)
    
    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
