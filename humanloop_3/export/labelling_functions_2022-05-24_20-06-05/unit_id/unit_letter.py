import re 
from hlprogrammatic_helpers import other_classes_regex

def unit_letter(row: Datapoint) -> List[Span]:
    """ 
    matches a flat like property
    followed by a space and letter
    followed by optional number or punctuation
    followed by a bunch of random stuff
    other_classes_regex+"s?\s([a-z]|([a-z](\d|\.|-)[a-z0-9\-\.]*))"
    """
    search_pattern = re.compile( other_classes_regex+r"s?\s([a-z](?=,| and)|[a-z](\d|\.|-)[a-z0-9\-\.]*)", flags=re.IGNORECASE)
    
    coords = [(m.span(2)[0], m.span(2)[1]) for m in re.finditer(search_pattern, row.text)]

    return [
      Span(start=start, end=end)
      for start, end in coords
    ]
