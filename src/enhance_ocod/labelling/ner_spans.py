"""
Library-agnostic NER labelling functions for property address entity extraction.

This module provides a suite of functions that extract entity spans (start, end, label) from text, covering buildings, cities, postcodes, streets, units, and more. Each function is designed to be independent of specific frameworks and can be used in any weak supervision pipeline.

Expected input: an object (row) with a `.text` attribute (str), plus any subfolder-specific attributes (e.g. flat_tag).

Typical usage:
    - Import and call any function with your row object.
    - Or iterate all functions in the `lfs` list for batch labelling.

Returns: List of (start, end, label) tuples for each entity found.
"""

import re
from typing import List, Tuple, Any

from .ner_regex import (
    building_regex,
    multi_word_no_land,
    building_special,
    company_type_regex,
    starting_building,
    xx_to_yy_regex,
    road_regex,
    postcode_regex,
    city_regex,
    special_streets,
    welsh_streets,
    waygate_regex,
    other_classes_regex
)

# Type aliases
EntitySpan = Tuple[int, int, str]
EntityList = List[EntitySpan]

# Common regex patterns as constants
ROAD_TYPES = (
    r"road|street|avenue|close|drive|place|grove|way|view|lane|row|wharf|quay|"
    r"meadow|field|park|gardens|crescent|market|arcade|parade|approach|bank|bar|"
    r"boulevard|court|croft|edge|footway|gate|green|hill|island|junction|maltings|"
    r"mews|mount|passage|point|promenade|spinney|square|terrace|thoroughfare|villas|wood"
)

UNIT_TYPES = (
    r"garage|storage|store|shed|outbuilding|barn|workshop|office|studio|annexe|annex|"
    r"loft|cellar|basement|utility|plant|boiler|meter|cupboard|bin|cycle|pram|buggy|"
    r"cleaner|caretaker|porter|concierge|warden|lobby|entrance|hall|stair|lift|escalator|"
    r"corridor|passage|landing|balcony|terrace|veranda|roof|garden|yard|drive|path|lane|track"
)

COMMON_ENDINGS = r"(?=,| and|$|;|:|\()"
FLAT_UNIT_PATTERN = r"apartment|flat|penthouse|unit|plot" #This is used to match the unit type so the s is not used
UNIT_PATTERN = r"(?:flat|apartment|unit|penthouse|plot)s?" #used as part of a broader matching regex so the s is useful


BUILDING = "building_name"
STREET_NAME = "street_name"
STREET_NUMBER = "street_number"
NUMBER_FILTER ="number_filter"
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
    pattern = re.compile(
        multi_word_no_land + building_regex + r"\b(?=,| and|$|;|:|\(|\s\d+,)",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), BUILDING) for m in pattern.finditer(row.text)]


def building_special_fn(row: Any) -> EntityList:
    """Extract special building names using specialized patterns."""
    pattern = re.compile(
        multi_word_no_land + building_special + COMMON_ENDINGS,
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), BUILDING) for m in pattern.finditer(row.text)]


def company_name(row: Any) -> EntityList:
    """Extract company names based on company type suffixes."""
    pattern = re.compile(
        r"[a-z\(\)'\s&]+" + company_type_regex,
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), COMPANY) for m in pattern.finditer(row.text)]


def mill(row: Any) -> EntityList:
    """Extract mill building names."""
    pattern = re.compile(
        r"(?<=[0-9]|,|\))\s([a-z][a-z'\s]+mill)(?=,| and|$|;|:)",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], BUILDING) for m in pattern.finditer(row.text)]


def number_comma_building_fn(row: Any) -> EntityList:
    """Extract building names following unit numbers and commas."""
    pattern = re.compile(
        rf"{UNIT_PATTERN} (?:[a-z]|\d+[a-z0-9\-\.]*|{xx_to_yy_regex}),\s([a-z\s.']+)(?=,)",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], BUILDING) for m in pattern.finditer(row.text)]


def start_only_letters_fn(row: Any) -> EntityList:
    """Extract building names that start with letters only."""
    pattern = re.compile(starting_building + COMMON_ENDINGS, flags=re.IGNORECASE)
    return [(m.start(), m.end(), BUILDING) for m in pattern.finditer(row.text)]


def wharf_building(row: Any) -> EntityList:
    """Extract wharf and similar building types."""
    pattern = re.compile(
        rf"(([a-z']+\s)+(wharf|quay|croft|priory|maltings|locks|gardens))(?:,)(?=[0-9a-z'\s]+{road_regex})",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], BUILDING) for m in pattern.finditer(row.text)]


def word_building_number(row: Any) -> EntityList:
    """Extract block/building identifiers with numbers."""
    pattern = re.compile(r"(block|building) ([a-z]([0-9]?)|\d+|)(?=,)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), BUILDING) for m in pattern.finditer(row.text)]


# ============================================================================
# CITY LABELLING FUNCTIONS
# ============================================================================

def city_match2(row: Any) -> EntityList:
    """Extract city names from the end of addresses."""
    pattern = re.compile(
        rf".*,\s*([^,]*?)(?=(?:(\sand\s|\s?\(?{postcode_regex})[^,]*)?$)",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], CITY) for m in pattern.finditer(row.text)]


# ============================================================================
# NUMBER FILTER LABELLING FUNCTIONS
# ============================================================================

def number_filter(row: Any) -> EntityList:
    """Extract number filter terms like 'odd' or 'even'."""
    pattern = re.compile(r"(?<=\()\s?(odd|even)(?=s?\b)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), NUMBER_FILTER) for m in pattern.finditer(row.text)]


# ============================================================================
# POSTCODE LABELLING FUNCTIONS
# ============================================================================

def postcodes(row: Any) -> EntityList:
    """Extract postcodes from text."""
    pattern = re.compile(postcode_regex, flags=re.IGNORECASE)
    return [(m.start(), m.end(), POSTCODE) for m in pattern.finditer(row.text)]


# ============================================================================
# STREET NAME LABELLING FUNCTIONS
# ============================================================================

def knightsbridge_road(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    pattern = re.compile(r"(?<=\d\s)knightsbridge(?=,)")
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def meadows_regex(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    pattern = re.compile(r"(?<=[0-9]\s)[a-z][a-z'\s]*meadow(s)?(?=,| and|$|;|:)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def number_space_multi_words_roadtitle_fn(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    pattern = re.compile(r"(?<=\d\s)(\b[a-z]+\s){2,5}("+road_regex+")")
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def park_roads(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    if getattr(row, 'commercial_park_tag', True):
        return []
    
    pattern = re.compile(r"\b[a-z][\sa-z']*park"+r"(?=,| and| \(|$)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def road_followed_city(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    pattern = re.compile(multi_word_no_land + road_regex +r"(?=\s"+city_regex+")", flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def road_names_basic(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    pattern = re.compile(multi_word_no_land+road_regex+r"(?=,| and| \(|$|;|:|\s"+city_regex+")", flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def side(row: Any) -> EntityList:
    """Extract roads which for some reason are called side."""
    pattern = re.compile(r"(new road|lodge|carr moor|lake|chase|thames|kennet|coppice|church|pool) side", flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def special_street_names(row: Any) -> EntityList:
    """Extract miscellaneous street names that do not fit the normal pattern."""
    special_streets = r"(pall mall|lower mall|haymarket|lower marsh|london wall|cheapside|eastcheap|piccadilly|aldwych|(the )?strand|point pleasant|bevis marks|old bailey|threelands|pendenza|castelnau|the old meadow|hortonwood|thoroughfare|navigation loop|turnberry|brentwood|hatton garden|whitehall|the quadrangle|green lanes|old jewry|st mary axe|minories|foxcover|meadow brook|daisy brook|north villas|south villas|march wall|millharbour|aztec west|trotwood|marlowes|petty france|petty cury|the quadrant|the spinney)"
    pattern = re.compile(special_streets + r"(?=,| and| \(|$)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def special_welsh(row: Any) -> EntityList:
    """Extract Welsh street names."""
    pattern = re.compile(welsh_streets, flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]

def the_dales(row: Any) -> EntityList:
    """Mark all matches of a regular expression within the document text as a span."""
    pattern = re.compile(r"(kirk|rye|avon|willow|darely |glen|thrush|nidder|moorside |arkengarth|wester|deep|fern|grise|common|moss)dale(?=,| and|$)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


def wharf_road(row: Any) -> EntityList:
    """Extract wharf and similar road types."""
    pattern = re.compile(r"("+multi_word_no_land+r"\b(wharf|quay(s)?|approach|parade|field(s)?|croft|priory|maltings|locks|gardens))(?:,)(?![0-9a-z'\s]+"+road_regex+")", flags=re.IGNORECASE)
    return [(m.span(1)[0], m.span(1)[1], STREET_NAME) for m in pattern.finditer(row.text)]


def words_waygate_fn(row: Any) -> EntityList:
    """Extract multi-word waygate patterns."""
    pattern = re.compile(multi_word_no_land+waygate_regex+r"(?=,| and| \()")
    return [(m.start(), m.end(), STREET_NAME) for m in pattern.finditer(row.text)]


# ============================================================================
# STREET NUMBER LABELLING FUNCTIONS
# ============================================================================

def adjective_space_number_words_roadtitle_fn(row: Any) -> EntityList:
    """Extract numbers following prepositions (at/on/in/etc.) before road-like words."""
    road_regex2 = r"(\s([a-z])+('s)?)+(\s"+road_regex+r")?"
    pattern = re.compile(r"(?:\b(?:at|on|in|adjoining|of|above|off|and|being|next to)\s)(\d+[a-z]?)(?="+road_regex2+r")")
    return [(m.span(1)[0], m.span(1)[1], STREET_NUMBER) for m in pattern.finditer(row.text)]


def and_space_n(row: Any) -> EntityList:
    """Extract numbers following 'and ' (only for non-flat-tagged rows)."""
    if getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(?<=and\s)(\d+[a-z]?)\b")
    return [(m.start(), m.end(), STREET_NUMBER) for m in pattern.finditer(row.text)]


def begins_with_number(row: Any) -> EntityList:
    """Extract number at beginning of text (only for non-flat-tagged rows)."""
    if getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"^(\d+([a-z])?)\b(?!(\s?(to|-)\s?)(\d+)\b)")
    match = pattern.search(row.text)
    return [(match.start(), match.end(), STREET_NUMBER)] if match else []


def comma_space_number_words_roadtitle_fn(row: Any) -> EntityList:
    """Extract numbers following comma and space before road-like words."""
    road_regex2 = r"(\s([a-z])+('s)?)+(\s"+road_regex+r")?"
    pattern = re.compile(r"(?<=,\s)\d+[a-z]?(?="+road_regex2+r")")
    return [(m.start(), m.end(), STREET_NUMBER) for m in pattern.finditer(row.text)]


def no_road_near(row: Any) -> EntityList:
    """Extract numbers following prepositions before punctuation (only for non-flat-tagged rows)."""
    if getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(at|on|in|adjoining|of|off|and|to|being|,)\s(\d+[a-z]?)\b(?=,| and|;|:|\s\()", flags=re.IGNORECASE)
    return [(m.span(2)[0], m.span(2)[1], STREET_NUMBER) for m in pattern.finditer(row.text)]


def xx_to_yy(row: Any) -> EntityList:
    """Extract number ranges (xx to yy) with conditional road context checking."""
    full_road_regex = "(([a-z'\s]+"+road_regex+r")|"+special_streets+ r")"
    full_regex = xx_to_yy_regex + r"(?="+ full_road_regex +r")"
    
    if not getattr(row, 'flat_tag', False):
        pattern = re.compile(xx_to_yy_regex, flags=re.IGNORECASE)
    else:
        pattern = re.compile(full_regex, flags=re.IGNORECASE)
    
    return [(m.start(), m.end(), STREET_NUMBER) for m in pattern.finditer(row.text)]


# ============================================================================
# UNIT ID LABELLING FUNCTIONS
# ============================================================================

def and_space_n_flats(row: Any) -> EntityList:
    """Extract flat numbers following 'and ' (only for flat-tagged rows)."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(?<=and\s)(\d+(\w{1})?)")
    return [(m.start(), m.end(), UNIT_ID) for m in pattern.finditer(row.text)]


def begins_with_number_flat(row: Any) -> EntityList:
    """Extract numbers at the beginning of flat-tagged text."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"^(\d+\w?)")
    match = pattern.search(row.text)
    return [(match.start(), match.end(), UNIT_ID)] if match else []


def number_before_building(row: Any) -> EntityList:
    """Extract numbers at start followed by building names (with optional comma)."""
    # Building names pattern - extend your existing building_regex with common building suffixes
    building_names =  building_regex 
    
    # Pattern: number at start + optional comma/space + building name
    pattern = re.compile(r"^([a-z]?\d+[a-z0-9\-\.]*),?\s+[^,]*?" + building_names, flags=re.IGNORECASE)
    match = pattern.search(row.text)
    return [(match.span(1)[0], match.span(1)[1], UNIT_ID)] if match else []

def carpark_id_fn(row: Any) -> EntityList:
    """Extract car park identifiers from garage/parking space descriptions."""
    pattern = re.compile(r"^(the )?(garage(s)?( space)?|parking(\s)?space|parking space(s)?|car park(ing)?( space))\s([a-z0-9\-\.]+\b)", flags=re.IGNORECASE)
    #pattern = re.compile(r"^(?:the\s+)?(?:\w+\s+)?(?:garage(?:s)?(?:\s+space)?|parking\s+space(?:s)?|car\s+park(?:ing)?(?:\s+space)?)\s+([a-z0-9\-\.]+)\b", flags=re.IGNORECASE)
    return [(m.span(9)[0], m.span(9)[1], UNIT_ID) for m in pattern.finditer(row.text)]


def comma_space_number_comma_flat(row: Any) -> EntityList:
    """Extract flat numbers between commas (only for flat-tagged rows)."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(?<=,\s)(\d+([a-z])?)(?=,)")
    return [(m.start(), m.end(), UNIT_ID) for m in pattern.finditer(row.text)]


def flat_letter(row: Any) -> EntityList:
    """Extract letter-based flat identifiers after apartment/flat/penthouse."""
    pattern = re.compile(UNIT_PATTERN + r"\s([a-z][0-9\.\-]*)(?=,)", flags=re.IGNORECASE)
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
    pattern = re.compile(UNIT_PATTERN + r"\s([a-z](\d|\.|-)[a-z0-9\-\.]*)", flags=re.IGNORECASE)
    return [(m.span(2)[0], m.span(2)[1], UNIT_ID) for m in pattern.finditer(row.text)]

def flat_number(row: Any) -> EntityList:
    """Extract flat numbers following property types (apartment|flat|penthouse)."""
    pattern = re.compile(UNIT_PATTERN + r"\s(\d+[a-z0-9\-\.]*)", flags=re.IGNORECASE)
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
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(?<!(-))\s(\d+)(?=\sand)", flags=re.IGNORECASE)
    return [(m.span(2)[0], m.span(2)[1], UNIT_ID) for m in pattern.finditer(row.text)]


def unit_id_fn(row: Any) -> EntityList:
    """Extract unit IDs following property class patterns."""
    pattern = re.compile(r"(the )?" + other_classes_regex + r"(s)?\s(\d[a-z0-9\-\.]*)", flags=re.IGNORECASE)
    return [(m.span(4)[0], m.span(4)[1], UNIT_ID) for m in pattern.finditer(row.text)]


def unit_letter(row: Any) -> EntityList:
    """Extract unit letters following property class patterns."""
    pattern = re.compile(other_classes_regex + r"s?\s([a-z](?=,| and)|[a-z](\d|\.|-)[a-z0-9\-\.]*)", flags=re.IGNORECASE)
    return [(m.span(2)[0], m.span(2)[1], UNIT_ID) for m in pattern.finditer(row.text)]


def xx_to_yy_flat(row: Any) -> EntityList:
    """Extract number ranges (xx to yy) for flat-tagged rows only."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    full_regex = xx_to_yy_regex + r"(?!\s(being|-|[a-z'\s]+" + road_regex + r"|" + special_streets + "))"
    pattern = re.compile(full_regex, flags=re.IGNORECASE)
    return [(m.start(), m.end(), UNIT_ID) for m in pattern.finditer(row.text)]


# ============================================================================
# UNIT TYPE LABELLING FUNCTIONS
# ============================================================================

def flat_unit(row: Any) -> EntityList:
    """Extract flat/apartment/penthouse/unit/plot types (excluding 'being' context)."""
    pattern = re.compile(r"(apartment|flat|penthouse|unit|plot)\b(?!being)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), UNIT_TYPE) for m in pattern.finditer(row.text)]


def is_carpark(row: Any) -> EntityList:
    """Extract car park unit types."""
    pattern = re.compile(r"^(garage(s)?|parking(\s)?space|parking space(s)?|car park(ing)?)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), UNIT_TYPE) for m in pattern.finditer(row.text)]


def is_other_class(row: Any) -> EntityList:
    """Extract other unit class types."""
    pattern = re.compile(other_classes_regex, flags=re.IGNORECASE)
    return [(m.start(), m.end(), UNIT_TYPE) for m in pattern.finditer(row.text)]


# ============================================================================
# MASTER FUNCTION LIST
# ============================================================================

# Master list of all labelling functions
lfs = [
    # Building functions
    building_regex_fn,
    building_special_fn,
    #company_name,
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