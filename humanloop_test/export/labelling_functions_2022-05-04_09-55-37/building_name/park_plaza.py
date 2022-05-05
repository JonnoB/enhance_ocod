import re 

def park_plaza(row: Datapoint) -> List[Span]:
    """ 
    Mark all instances of a search string within
    the document text as a span 
    """
    search_term = "park plaza westminster bridge"
    
    starts = [m.start() for m in re.finditer(search_term, row.text)]

    return [
      Span(start=index, end=index + len(search_term))
      for index in starts
    ]
