import re
from hlprogrammatic_helpers import special_streets

def special_street_names(row: Datapoint) -> List[Span]:
    """
    A collection of miscallaneous street names that do not fit the normal pattern
    """

    #special_streets = r"(pall mall|lower mall|haymarket|lower marsh|london wall|cheapside|eastcheap|piccadilly|aldwych|(the )?strand|point pleasant|bevis marks|old bailey|threelands|pendenza|castelnau|the old meadow|hortonwood|thoroughfare|navigation loop|turnberry|brentwood|hatton garden|whitehall|the quadrangle|green lanes|old jewry|st mary axe|minories|foxcover|meadow brook|daisy brook|north villas|south villas|march wall|millharbour|aztec west|trotwood|marlowes|petty france|petty cury|the quadrant|the spinney)"

    search_pattern = re.compile(special_streets + r"(?=,| and| \(|$)", flags=re.IGNORECASE)

    coords = [(m.start(), m.end()) for m in re.finditer(search_pattern, row.text)]

    return [Span(start=start, end=end) for start, end in coords]
