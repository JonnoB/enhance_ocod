import re


def the_dales(row: Datapoint) -> List[Span]:
    """
    Mark all matches of a regular expression within
    the document text as a span
    """
    search_pattern = re.compile(r"(kirk|rye|avon|willow|darely |glen|thrush|nidder|moorside |arkengarth|wester|deep|fern|grise|common|moss)dale(?=,| and|$)", flags=re.IGNORECASE)

    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
