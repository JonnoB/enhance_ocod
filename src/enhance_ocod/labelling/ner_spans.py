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
FLAT_UNIT_PATTERN = r"apartment|flat|penthouse"
UNIT_PATTERN = r"(?:flat|apartment|unit)s?"


# ============================================================================
# BUILDING LABELLING FUNCTIONS
# ============================================================================

def building_regex_fn(row: Any) -> EntityList:
    """Extract building names using standard building regex patterns."""
    pattern = re.compile(
        multi_word_no_land + building_regex + r"\b(?=,| and|$|;|:|\(|\s\d+,)",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "BUILDING") for m in pattern.finditer(row.text)]


def building_special_fn(row: Any) -> EntityList:
    """Extract special building names using specialized patterns."""
    pattern = re.compile(
        multi_word_no_land + building_special + COMMON_ENDINGS,
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "BUILDING") for m in pattern.finditer(row.text)]


def company_name(row: Any) -> EntityList:
    """Extract company names based on company type suffixes."""
    pattern = re.compile(
        r"[a-z\(\)'\s&]+" + company_type_regex,
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "COMPANY") for m in pattern.finditer(row.text)]


def mill(row: Any) -> EntityList:
    """Extract mill building names."""
    pattern = re.compile(
        r"(?<=[0-9]|,|\))\s([a-z][a-z'\s]+mill)(?=,| and|$|;|:)",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], "BUILDING") for m in pattern.finditer(row.text)]


def number_comma_building_fn(row: Any) -> EntityList:
    """Extract building names following unit numbers and commas."""
    pattern = re.compile(
        rf"{UNIT_PATTERN} (?:[a-z]|\d+[a-z0-9\-\.]*|{xx_to_yy_regex}),\s([a-z\s.']+)(?=,)",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], "BUILDING") for m in pattern.finditer(row.text)]


def start_only_letters_fn(row: Any) -> EntityList:
    """Extract building names that start with letters only."""
    pattern = re.compile(starting_building + COMMON_ENDINGS, flags=re.IGNORECASE)
    return [(m.start(), m.end(), "BUILDING") for m in pattern.finditer(row.text)]


def wharf_building(row: Any) -> EntityList:
    """Extract wharf and similar building types."""
    pattern = re.compile(
        rf"(([a-z']+\s)+(wharf|quay|croft|priory|maltings|locks|gardens))(?:,)(?=[0-9a-z'\s]+{road_regex})",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], "BUILDING") for m in pattern.finditer(row.text)]


def word_building_number(row: Any) -> EntityList:
    """Extract block/building identifiers with numbers."""
    pattern = re.compile(r"(block|building) ([a-z]([0-9]?)|\d+|)(?=,)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), "BUILDING") for m in pattern.finditer(row.text)]


# ============================================================================
# CITY LABELLING FUNCTIONS
# ============================================================================

def city_match2(row: Any) -> EntityList:
    """Extract city names from the end of addresses."""
    pattern = re.compile(
        rf".*,\s*([^,]*?)(?=(?:(\sand\s|\s?\(?{postcode_regex})[^,]*)?$)",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], "CITY") for m in pattern.finditer(row.text)]


# ============================================================================
# NUMBER FILTER LABELLING FUNCTIONS
# ============================================================================

def number_filter(row: Any) -> EntityList:
    """Extract number filter terms like 'odd' or 'even'."""
    pattern = re.compile(r"(?<=\()\s?(odd|even)(?=s?\b)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), "NUMBER_FILTER") for m in pattern.finditer(row.text)]


# ============================================================================
# POSTCODE LABELLING FUNCTIONS
# ============================================================================

def postcodes(row: Any) -> EntityList:
    """Extract postcodes from text."""
    pattern = re.compile(postcode_regex, flags=re.IGNORECASE)
    return [(m.start(), m.end(), "POSTCODE") for m in pattern.finditer(row.text)]


# ============================================================================
# STREET NAME LABELLING FUNCTIONS
# ============================================================================

def knightsbridge_road(row: Any) -> EntityList:
    """Extract Knightsbridge road references."""
    pattern = re.compile(r"(?<=\d\s)knightsbridge(?=,)")
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def meadows_regex(row: Any) -> EntityList:
    """Extract meadow street names."""
    pattern = re.compile(
        r"(?<=[0-9]\s)[a-z][a-z'\s]*meadow(s)?(?=,| and|$|;|:)",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def number_space_multi_words_roadtitle_fn(row: Any) -> EntityList:
    """Extract multi-word street names following numbers."""
    pattern = re.compile(rf"(?<=\d\s)(\b[a-z]+\s){{2,5}}({road_regex})")
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def park_roads(row: Any) -> EntityList:
    """Extract park road names, excluding commercial parks."""
    pattern = re.compile(r"\b[a-z][\sa-z']*park(?=,| and| \(|$)", flags=re.IGNORECASE)
    
    if getattr(row, 'commercial_park_tag', False):
        return []
    
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def road_followed_city(row: Any) -> EntityList:
    """Extract road names followed by city names."""
    pattern = re.compile(
        multi_word_no_land + road_regex + rf"(?=\s{city_regex})",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def road_names_basic(row: Any) -> EntityList:
    """Extract basic road names with common endings."""
    pattern = re.compile(
        multi_word_no_land + road_regex + rf"(?=,| and| \(|$|;|:|\s{city_regex})",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def side(row: Any) -> EntityList:
    """Extract 'side' street names."""
    pattern = re.compile(
        r"(new road|lodge|carr moor|lake|chase|thames|kennet|coppice|church|pool) side",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def special_street_names(row: Any) -> EntityList:
    """Extract special street names from predefined list."""
    pattern = re.compile(special_streets + r"(?=,| and| \(|$)", flags=re.IGNORECASE)
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def special_welsh(row: Any) -> EntityList:
    """Extract Welsh street names."""
    pattern = re.compile(welsh_streets, flags=re.IGNORECASE)
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def the_dales(row: Any) -> EntityList:
    """Extract dale street names."""
    pattern = re.compile(
        r"(kirk|rye|avon|willow|darely |glen|thrush|nidder|moorside |arkengarth|"
        r"wester|deep|fern|grise|common|moss)dale(?=,| and|$)",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def welsh_pattern(row: Any) -> EntityList:
    """Extract Welsh street patterns (lon, llys, fford, clos)."""
    pattern = re.compile(
        r"\b(lon|llys|fford|clos)\s[a-z\-\s']+(?=,| and|$)",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


def wharf_road(row: Any) -> EntityList:
    """Extract wharf roads that are not followed by other road names."""
    pattern = re.compile(
        rf"({multi_word_no_land}\b(wharf|quay(s)?|approach|parade|field(s)?|croft|"
        rf"priory|maltings|locks|gardens))(?:,)(?![0-9a-z'\s]+{road_regex})",
        flags=re.IGNORECASE
    )
    return [(m.span(1)[0], m.span(1)[1], "STREET") for m in pattern.finditer(row.text)]


def words_waygate_fn(row: Any) -> EntityList:
    """Extract waygate street names."""
    pattern = re.compile(multi_word_no_land + waygate_regex + r"(?=,| and| \()")
    return [(m.start(), m.end(), "STREET") for m in pattern.finditer(row.text)]


# ============================================================================
# STREET NUMBER LABELLING FUNCTIONS
# ============================================================================

def adjective_space_number_words_roadtitle_fn(row: Any) -> EntityList:
    """Extract street numbers with adjectives and road titles."""
    pattern = re.compile(
        rf"(?<=\b)([a-z]+\s){{1,3}}(\d+)(\s[a-z]+)?(\s{ROAD_TYPES})",
        flags=re.IGNORECASE
    )
    return [(m.start(), m.end(), "STREET_NUMBER") for m in pattern.finditer(row.text)]


def and_space_n(row: Any) -> EntityList:
    """Extract numbers following 'and '."""
    pattern = re.compile(r"and\s(\d+)")
    return [(m.start(1), m.end(1), "STREET_NUMBER") for m in pattern.finditer(row.text)]


def begins_with_number(row: Any) -> EntityList:
    """Extract numbers at the beginning of text."""
    pattern = re.compile(r"^(\d+)")
    return [(m.start(1), m.end(1), "STREET_NUMBER") for m in pattern.finditer(row.text)]


def comma_space_number_words_roadtitle_fn(row: Any) -> EntityList:
    """Extract street numbers after commas with road titles."""
    pattern = re.compile(
        rf",\s(\d+)(\s[a-z]+){{1,3}}(\s{ROAD_TYPES})",
        flags=re.IGNORECASE
    )
    return [(m.start(1), m.end(1), "STREET_NUMBER") for m in pattern.finditer(row.text)]


def no_road_near(row: Any) -> EntityList:
    """Extract 'no road near' patterns."""
    pattern = re.compile(r"\b(no\sroad\snear)\b", flags=re.IGNORECASE)
    return [(m.start(), m.end(), "STREET_NUMBER") for m in pattern.finditer(row.text)]


def xx_to_yy(row: Any) -> EntityList:
    """Extract number ranges (xx to yy)."""
    pattern = re.compile(xx_to_yy_regex)
    return [(m.start(), m.end(), "STREET_NUMBER") for m in pattern.finditer(row.text)]


# ============================================================================
# UNIT ID LABELLING FUNCTIONS
# ============================================================================

def and_space_n_flats(row: Any) -> EntityList:
    """Extract flat numbers following 'and ' (only for flat-tagged rows)."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(?<=and\s)(\d+(\w{1})?)")
    return [(m.start(), m.end(), "UNIT_ID") for m in pattern.finditer(row.text)]


def begins_with_number_flat(row: Any) -> EntityList:
    """Extract flat numbers at the beginning (only for flat-tagged rows)."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"^(\d+\w?)")
    match = pattern.search(row.text)
    return [(match.start(), match.end(), "UNIT_ID")] if match else []


def carpark_id_fn(row: Any) -> EntityList:
    """Extract car park/garage space IDs."""
    pattern = re.compile(
        r"^(the )?(garage(s)?( space)?|parking(\s)?space|parking space(s)?|"
        r"car park(ing)?( space))\s([a-z0-9\-\.]+\b)",
        flags=re.IGNORECASE
    )
    return [(m.span(9)[0], m.span(9)[1], "UNIT_ID") for m in pattern.finditer(row.text)]


def comma_space_number_comma_flat(row: Any) -> EntityList:
    """Extract flat unit IDs between commas (only for flat-tagged rows)."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(?<=,\s)(\d+([a-z])?)(?=,)")
    return [(m.start(), m.end(), "UNIT_ID") for m in pattern.finditer(row.text)]


def flat_letter(row: Any) -> EntityList:
    """Extract letter-based flat/apartment IDs."""
    pattern = re.compile(
        rf"({FLAT_UNIT_PATTERN})\s([a-z][0-9\.\-]*)(?=,)",
        flags=re.IGNORECASE
    )
    return [(m.span(2)[0], m.span(2)[1], "UNIT_ID") for m in pattern.finditer(row.text)]


def flat_letter_complex(row: Any) -> EntityList:
    """Extract complex letter-based flat/apartment IDs."""
    pattern = re.compile(
        rf"({FLAT_UNIT_PATTERN})\s([a-z](\d|\.|-)[a-z0-9\-\.]*)",
        flags=re.IGNORECASE
    )
    return [(m.span(2)[0], m.span(2)[1], "UNIT_ID") for m in pattern.finditer(row.text)]


def flat_number(row: Any) -> EntityList:
    """Extract number-based flat/apartment IDs."""
    pattern = re.compile(
        rf"({FLAT_UNIT_PATTERN})\s(\d+[a-z0-9\-\.]*)",
        flags=re.IGNORECASE
    )
    return [(m.span(2)[0], m.span(2)[1], "UNIT_ID") for m in pattern.finditer(row.text)]


def number_space_and_flat(row: Any) -> EntityList:
    """Extract flat numbers before 'and' (only for flat-tagged rows)."""
    if not getattr(row, 'flat_tag', False):
        return []
    
    pattern = re.compile(r"(?<!(-))\s(\d+)(?=\sand)", flags=re.IGNORECASE)
    return [(m.span(2)[0], m.span(2)[1], "UNIT_ID") for m in pattern.finditer(row.text)]


def unit_id_fn(row: Any) -> EntityList:
    """Extract general unit IDs."""
    pattern = re.compile(r"unit\s([a-z0-9\-\.]+)", flags=re.IGNORECASE)
    return [(m.span(1)[0], m.span(1)[1], "UNIT_ID") for m in pattern.finditer(row.text)]


def unit_letter(row: Any) -> EntityList:
    """Extract single letter unit IDs."""
    pattern = re.compile(r"unit\s([a-z])", flags=re.IGNORECASE)
    return [(m.span(1)[0], m.span(1)[1], "UNIT_ID") for m in pattern.finditer(row.text)]


def xx_to_yy_flat(row: Any) -> EntityList:
    """Extract unit ID ranges (xx to yy)."""
    pattern = re.compile(xx_to_yy_regex)
    return [(m.start(), m.end(), "UNIT_ID") for m in pattern.finditer(row.text)]


# ============================================================================
# UNIT TYPE LABELLING FUNCTIONS
# ============================================================================

def flat_unit(row: Any) -> EntityList:
    """Extract flat/apartment/penthouse unit types."""
    pattern = re.compile(FLAT_UNIT_PATTERN, flags=re.IGNORECASE)
    return [(m.start(), m.end(), "UNIT_TYPE") for m in pattern.finditer(row.text)]


def is_carpark(row: Any) -> EntityList:
    """Extract car park unit types."""
    pattern = re.compile(r"car\s?park", flags=re.IGNORECASE)
    return [(m.start(), m.end(), "UNIT_TYPE") for m in pattern.finditer(row.text)]


def is_other_class(row: Any) -> EntityList:
    """Extract other unit class types."""
    pattern = re.compile(UNIT_TYPES, flags=re.IGNORECASE)
    return [(m.start(), m.end(), "UNIT_TYPE") for m in pattern.finditer(row.text)]


# ============================================================================
# MASTER FUNCTION LIST
# ============================================================================

# Master list of all labelling functions
lfs = [
    # Building functions
    building_regex_fn,
    building_special_fn,
    company_name,
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
    welsh_pattern,
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