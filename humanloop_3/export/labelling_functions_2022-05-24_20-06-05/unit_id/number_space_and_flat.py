import re
from hlprogrammatic_helpers import road_regex

def number_space_and_flat(row: Datapoint) -> List[Span]:
    """ 
    looks for a number followed by a space followed by "and"
    """
    if row.flat_tag ==True:

      search_pattern = re.compile(r"(?<!(-))\s(\d+)(?=\sand)", flags=re.IGNORECASE)
    
      coords = [(m.span(2)[0], m.span(2)[1]) for m in re.finditer(search_pattern, row.text)]

    else:

       coords = []

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
