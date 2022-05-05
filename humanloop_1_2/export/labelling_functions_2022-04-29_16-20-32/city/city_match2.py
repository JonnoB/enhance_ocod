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
    search_pattern = re.compile(r".*,\s*([^,]*?)(?=(?:(\sand\s|\s?\(?\b[a-z]{1,2}\d[a-z0-9]?\s\d[a-z]{2}\b)[^,]*)?$)", flags=re.IGNORECASE)

    coords = [(m.group(1).start(), m.group(1).end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
