import re
from hlprogrammatic_helpers import postcode_regex

def city_match2(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    ([a-z\s]+)\b
    r"(?<=\b)(?:[a-z][a-z.\-\s][a-z](?!(?:and)\b))*"
    r"(?<=(,\s))[a-z][a-z\s\-.]+[a-z]\b(?=(\s)?(\()?"+ postcode_regex +")" original
    r",\s*((?:(?!\s(and\s|\s?\(?"+postcode_regex+r"))[^,])*)(?=[^,]*$)"
    """
    search_pattern = re.compile(r".*,\s*([^,]*?)(?=(?:(\sand\s|\s?\(?"+postcode_regex+r")[^,]*)?$)", flags=re.IGNORECASE)

    coords = [(m.span(1)[0], m.span(1)[1]) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
