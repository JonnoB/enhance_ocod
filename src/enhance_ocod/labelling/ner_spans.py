"""
Library-agnostic NER labelling functions for property address entity extraction.
"""

from typing import List, Tuple, Any
from .ner_regex import PATTERNS  

# Type aliases
EntitySpan = Tuple[int, int, str]
EntityList = List[EntitySpan]

# Labels
BUILDING = "building_name"
STREET_NAME = "street_name"
STREET_NUMBER = "street_number"
NUMBER_FILTER = "number_filter"
UNIT_ID = "unit_id"
UNIT_TYPE = "unit_type"
CITY = "city"
POSTCODE = "postcode"
COMPANY = "company"

# ============================================================================
# BUILDING LABELLING FUNCTIONS
# ============================================================================

def building_regex_fn(row: Any) -> EntityList:
    """Extract building names using standard building regex patterns."""
    return [(m.start(), m.end(), BUILDING) for m in PATTERNS.BUILDING_REGEX.finditer(row.text)]

def building_special_fn(row: Any) -> EntityList:
    """Extract special building names using specialized patterns."""
    return [(m.start(), m.end(), BUILDING) for m in PATTERNS.BUILDING_SPECIAL.finditer(row.text)]

def company_name(row: Any) -> EntityList:
    """Extract company names based on company type suffixes."""
    return [(m.start(), m.end(), COMPANY) for m in PATTERNS.COMPANY_NAME.finditer(row.text)]

def mill(row: Any) -> EntityList:
    """Extract mill building names."""
    return [(m.span(1)[0], m.span(1)[1], BUILDING) for m in PATTERNS.MILL.finditer(row.text)]

def number_comma_building_fn(row: Any) -> EntityList:
    """Extract building names following unit numbers and commas."""
    return [(m.span(1)[0], m.span(1)[1], BUILDING) for m in PATTERNS.NUMBER_COMMA_BUILDING.finditer(row.text)]

def start_only_letters_fn(row: Any) -> EntityList:
    """Extract building names that start with letters only."""
    return [(m.start(), m.end(), BUILDING) for m in PATTERNS.START_ONLY_LETTERS.finditer(row.text)]

def wharf_building(row: Any) -> EntityList:
    """Extract wharf and similar building types."""
    return [(m.span(1)[0], m.span(1)[1], BUILDING) for m in PATTERNS.WHARF_BUILDING.finditer(row.text)]

def word_building_number(row: Any) -> EntityList:
    """Extract block/building identifiers with numbers."""
    return [(m.start(), m.end(), BUILDING) for m in PATTERNS.WORD_BUILDING_NUMBER.finditer(row.text)]

# ============================================================================
# CITY LABELLING FUNCTIONS
# ============================================================================

def city_match2(row: Any) -> EntityList:
    """Extract city names from the end of addresses."""
    return [(m.span(1)[0], m.span(1)[1], CITY) for m in PATTERNS.CITY_MATCH.finditer(row.text)]

# ============================================================================
# NUMBER FILTER LABELLING FUNCTIONS
# ============================================================================

def number_filter(row: Any) -> EntityList:
    """Extract number filter terms like 'odd' or 'even'."""
    return [(m.start(), m.end(), NUMBER_FILTER) for m in PATTERNS.NUMBER_FILTER.finditer(row.text)]

# ============================================================================
# POSTCODE LABELLING FUNCTIONS
# ============================================================================

def postcodes(row: Any) -> EntityList:
    """Extract postcodes from text."""
    return [(m.start(), m.end(), POSTCODE) for m in PATTERNS.POSTCODE.finditer(row.text)]


# ============================================================================
# STREET NAME LABELLING FUNCTIONS
# ============================================================================


def knightsbridge_road(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.KNIGHTSBRIDGE_ROAD.finditer(row.text)]



def meadows_regex(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.MEADOWS_REGEX.finditer(row.text)]


def number_space_multi_words_roadtitle_fn(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.NUMBER_SPACE_MULTI_WORDS_ROADTITLE.finditer(row.text)]


def park_roads(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    if getattr(row, "commercial_park_tag", True):
        return []

    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.PARK_ROADS.finditer(row.text)]


def road_followed_city(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""

    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.ROAD_FOLLOWED_CITY.finditer(row.text)]


def road_names_basic(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""

    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.ROAD_NAMES_BASIC.finditer(row.text)]


def side(row: Any) -> EntityList:
    """Extract roads which for some reason are called side."""

    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.SIDE.finditer(row.text)]


def special_street_names(row: Any) -> EntityList:
    """Extract miscellaneous street names that do not fit the normal pattern."""

    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.SPECIAL_STREET_NAMES.finditer(row.text)]


def special_welsh(row: Any) -> EntityList:
    """Extract Welsh street names."""
    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.SPECIAL_WELSH.finditer(row.text)]


def the_dales(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""

    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.THE_DALES.finditer(row.text)]


def wharf_road(row: Any) -> EntityList:
    """Extract wharf and similar road types."""

    return [(m.span(1)[0], m.span(1)[1], STREET_NAME) for m in PATTERNS.WHARF_ROAD.finditer(row.text)]


def words_waygate_fn(row: Any) -> EntityList:
    """Extract multi-word waygate patterns."""
    return [(m.start(), m.end(), STREET_NAME) for m in PATTERNS.WORDS_WAYGATE.finditer(row.text)]


# ============================================================================
# STREET NUMBER LABELLING FUNCTIONS
# ============================================================================


def adjective_space_number_words_roadtitle_fn(row: Any) -> EntityList:
    """Extract numbers following prepositions (at/on/in/etc.) before road-like words."""
    return [
        (m.span(1)[0], m.span(1)[1], STREET_NUMBER) for m in PATTERNS.ADJECTIVE_SPACE_NUMBER_WORDS_ROADTITLE.finditer(row.text)
    ]


def and_space_n(row: Any) -> EntityList:
    """Extract numbers following 'and ' (only for non-flat-tagged rows)."""
    if getattr(row, "flat_tag", False):
        return []

    return [(m.start(), m.end(), STREET_NUMBER) for m in PATTERNS.AND_SPACE_N.finditer(row.text)]


def begins_with_number(row: Any) -> EntityList:
    """Extract number at beginning of text (only for non-flat-tagged rows)."""
    if getattr(row, "flat_tag", False):
        return []

    match = PATTERNS.BEGINS_WITH_NUMBER.search(row.text)
    return [(match.start(), match.end(), STREET_NUMBER)] if match else []


def comma_space_number_words_roadtitle_fn(row: Any) -> EntityList:
    """Extract numbers following comma and space before road-like words."""

    return [(m.start(), m.end(), STREET_NUMBER) for m in PATTERNS.COMMA_SPACE_NUMBER_WORDS_ROADTITLE.finditer(row.text)]

def no_road_near(row: Any) -> EntityList:
    """Extract numbers following prepositions before punctuation (only for non-flat-tagged rows)."""
    if getattr(row, "flat_tag", False):
        return []
    return [
        (m.span(2)[0], m.span(2)[1], STREET_NUMBER) for m in PATTERNS.NO_ROAD_NEAR.finditer(row.text)
    ]


def xx_to_yy(row: Any) -> EntityList:
    """Extract number ranges (xx to yy) with conditional road context checking."""

    if not getattr(row, "flat_tag", False):
        pattern = PATTERNS.XX_TO_YY
    else:
        pattern = PATTERNS.XX_TO_YY_WITH_ROAD

    return [(m.start(), m.end(), STREET_NUMBER) for m in pattern.finditer(row.text)]


# ============================================================================
# UNIT ID LABELLING FUNCTIONS
# ============================================================================


def and_space_n_flats(row: Any) -> EntityList:
    """Extract flat numbers following 'and ' (only for flat-tagged rows)."""
    if not getattr(row, "flat_tag", False):
        return []
    
    return [(m.start(), m.end(), UNIT_ID) for m in PATTERNS.AND_SPACE_N_FLATS.finditer(row.text)]


def begins_with_number_flat(row: Any) -> EntityList:
    """Extract numbers at the beginning of flat-tagged text."""
    if not getattr(row, "flat_tag", False):
        return []

    match = PATTERNS.BEGINS_WITH_NUMBER_FLAT
    return [(match.start(), match.end(), UNIT_ID)] if match else []


def number_before_building(row: Any) -> EntityList:
    """Extract numbers at start followed by building names (with optional comma)."""
    # Building names pattern - extend your existing building_regex with common building suffixes
    match = PATTERNS.NUMBER_BEFORE_BUILDING.search(row.text)
    return [(match.span(1)[0], match.span(1)[1], UNIT_ID)] if match else []


def carpark_id_fn(row: Any) -> EntityList:
    """Extract car park identifiers from garage/parking space descriptions."""
    return [(m.span(9)[0], m.span(9)[1], UNIT_ID) for m in PATTERNS.CARPARK_ID.finditer(row.text)]


def comma_space_number_comma_flat(row: Any) -> EntityList:
    """Extract flat numbers between commas (only for flat-tagged rows)."""
    if not getattr(row, "flat_tag", False):
        return []

    return [(m.start(), m.end(), UNIT_ID) for m in PATTERNS.COMMA_SPACE_NUMBER_COMMA_FLAT.finditer(row.text)]


def flat_letter(row: Any) -> EntityList:
    """Extract letter-based flat identifiers after apartment/flat/penthouse."""
    pattern = PATTERNS.FLAT_LETTER
    matches = []
    for m in pattern.finditer(row.text):
        if len(m.groups()) >= 1:  # Ensure group 1 exists
            # Calculate the position of just the letter part
            full_start = m.start()
            full_text = m.group(0)
            letter_text = m.group(1)
            letter_start = full_start + full_text.find(letter_text)
            letter_end = letter_start + len(letter_text)
            matches.append((letter_start, letter_end, UNIT_ID))
    return matches


def flat_letter_complex(row: Any) -> EntityList:
    """Extract complex letter-based flat identifiers after apartment/flat/penthouse."""

    return [(m.span(2)[0], m.span(2)[1], UNIT_ID) for m in PATTERNS.FLAT_LETTER_COMPLEX.finditer(row.text)]


def flat_number(row: Any) -> EntityList:
    """Extract flat numbers following property types (apartment|flat|penthouse)."""
    pattern = PATTERNS.FLAT_NUMBER
    matches = []
    for m in pattern.finditer(row.text):
        if len(m.groups()) >= 1:  # Ensure group 1 exists
            # Calculate the position of just the number part
            full_start = m.start()
            full_text = m.group(0)
            number_text = m.group(1)
            number_start = full_start + full_text.find(number_text)
            number_end = number_start + len(number_text)
            matches.append((number_start, number_end, UNIT_ID))
    return matches


def number_space_and_flat(row: Any) -> EntityList:
    """Extract numbers followed by space and 'and' (only for flat-tagged rows)."""
    if not getattr(row, "flat_tag", False):
        return []

    return [(m.span(2)[0], m.span(2)[1], UNIT_ID) for m in PATTERNS.NUMBER_SPACE_AND_FLAT.finditer(row.text)]


def unit_id_fn(row: Any) -> EntityList:
    """Extract unit IDs following property class patterns."""
    return [(m.span(4)[0], m.span(4)[1], UNIT_ID) for m in PATTERNS.UNIT_ID.finditer(row.text)]


def unit_letter(row: Any) -> EntityList:
    """Extract unit letters following property class patterns."""
    return [(m.span(2)[0], m.span(2)[1], UNIT_ID) for m in PATTERNS.UNIT_LETTER.finditer(row.text)]


def xx_to_yy_flat(row: Any) -> EntityList:
    """Extract number ranges (xx to yy) for flat-tagged rows only."""
    if not getattr(row, "flat_tag", False):
        return []

    return [(m.start(), m.end(), UNIT_ID) for m in PATTERNS.XX_TO_YY_FLAT.finditer(row.text)]


# ============================================================================
# UNIT TYPE LABELLING FUNCTIONS
# ============================================================================


def flat_unit(row: Any) -> EntityList:
    """Extract flat/apartment/penthouse/unit/plot types (excluding 'being' context)."""

    return [(m.start(), m.end(), UNIT_TYPE) for m in PATTERNS.FLAT_UNIT.finditer(row.text)]


def is_carpark(row: Any) -> EntityList:
    """Extract car park unit types."""

    return [(m.start(), m.end(), UNIT_TYPE) for m in PATTERNS.IS_CARPARK.finditer(row.text)]


def is_other_class(row: Any) -> EntityList:
    """Extract other unit class types."""
    return [(m.start(), m.end(), UNIT_TYPE) for m in PATTERNS.IS_OTHER_CLASS.finditer(row.text)]


# ============================================================================
# MASTER FUNCTION LIST
# ============================================================================

# Master list of all labelling functions
lfs = [
    # Building functions
    building_regex_fn,
    building_special_fn,
    # company_name,
    mill,
    number_comma_building_fn,
    start_only_letters_fn,
    wharf_building,
    word_building_number,
    # City functions
    city_match2,
    # Number filter functions
    number_filter,
    # Postcode functions
    postcodes,
    # Street name functions
    knightsbridge_road,
    meadows_regex,
    number_space_multi_words_roadtitle_fn,
    park_roads,
    road_followed_city,
    road_names_basic,
    side,
    special_street_names,
    special_welsh,
    the_dales,
    special_welsh,
    wharf_road,
    words_waygate_fn,
    # Street number functions
    adjective_space_number_words_roadtitle_fn,
    and_space_n,
    begins_with_number,
    comma_space_number_words_roadtitle_fn,
    no_road_near,
    xx_to_yy,
    # Unit ID functions
    and_space_n_flats,
    begins_with_number_flat,
    number_before_building,
    carpark_id_fn,
    comma_space_number_comma_flat,
    flat_letter,
    flat_letter_complex,
    flat_number,
    number_space_and_flat,
    unit_id_fn,
    unit_letter,
    xx_to_yy_flat,
    # Unit type functions
    flat_unit,
    is_carpark,
    is_other_class,
]
