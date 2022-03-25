import re

def begins_with_number_flat(row: Datapoint) -> List[Span]:
    """ 
    Mark the first match of a regex pattern within
    the document text as a span 
    """

    #road_regex = "court"
    if row.flat_tag == True:
       # search_pattern = re.compile("^(\d+)(?!(([a-z])+('s)?)+(\s"+ road_regex +")" )
        search_pattern = re.compile("^(\d+\w)" )
        match = re.search(search_pattern, row.text)
    else:
        match = None

    if match:
        return Span(start=match.start(), end=match.end())
