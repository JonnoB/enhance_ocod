import re
from hlprogrammatic_helpers import road_regex
from hlprogrammatic_helpers import xx_to_yy_regex
from hlprogrammatic_helpers import special_streets

def xx_to_yy(row: Datapoint) -> List[Span]:
    """ 
    Mark all matches of a regular expression within
    the document text as a span
    """
    full_road_regex  = "(([a-z'\s]+"+road_regex+r")|"+special_streets+ r")"
    full_regex = xx_to_yy_regex + r"(?="+ full_road_regex +r")"
    if row.flat_tag ==False:
    #prevents flat groups being marked as road addresses
      search_pattern = re.compile(xx_to_yy_regex, flags=re.IGNORECASE)
    
      coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]
    else: 
    #this relaxes the rule as any xx to yy must be a street address if not a flat
      search_pattern = re.compile(full_regex, flags=re.IGNORECASE)
      coords = coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]
    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
