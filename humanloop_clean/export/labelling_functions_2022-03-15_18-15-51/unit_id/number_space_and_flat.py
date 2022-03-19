import re
from hlprogrammatic_helpers import road_regex

def number_space_and_flat(row: Datapoint) -> List[Span]:
    """ 
    looks for a number followed by a space followed by "and"
    """
    if row.flat_tag ==True:

      search_pattern = re.compile(r"(?<!(-))(\b\d+)(?=\sand)", flags=re.IGNORECASE)
    
      coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    else:

       coords = [(m.start(), m.end()) for m in re.finditer("load of old trousers", row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
