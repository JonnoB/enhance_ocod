import re
from hlprogrammatic_helpers import road_regex
from hlprogrammatic_helpers import xx_to_yy_regex

def xx_to_yy_flat(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """

    if row.flat_tag ==True:
      #adding in a not preceded by - prevents xx-yy-xx patterns
      #many random things like floors xx to yy being flats 0-36
      
      full_regex = xx_to_yy_regex + r"(?!(\sbeing|-|" +road_regex +r"))"
      search_pattern = re.compile(full_regex , flags=re.IGNORECASE)
    
      coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]
    else: 
      coords = coords = [(m.start(), m.end()) for m in re.finditer('there_aint_nothing_here', row.text)]
    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
