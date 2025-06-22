"""
Snorkel-compatible NER span-level labeling functions

Each function returns a list of span dictionaries or ABSTAIN.
Functions are grouped by their original subfolder for traceability.

Expected input: a DataPoint object with a `.text` attribute (str), plus any 
subfolder-specific attributes (e.g. flat_tag).
"""

import re
from typing import List, Dict, Any, Union
from snorkel.labeling import labeling_function
from snorkel.types import DataPoint

from ner_regex import (
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


# Constants for entity types
BUILDING = "BUILDING"
STREET = "STREET"
POSTCODE = "POSTCODE"
COMPANY = "COMPANY"
CITY = "CITY"
UNIT_ID = "UNIT_ID"
UNIT_TYPE = "UNIT_TYPE"
STREET_NUMBER = "STREET_NUMBER"
NUMBER_FILTER = "NUMBER_FILTER"

# Constants
ABSTAIN = -1

# Common regex patterns as constants
ROAD_TYPES = (
    r"road|street|avenue|close|drive|place|grove|way|view|lane|row|wharf|quay|"
    r"meadow|field|park|gardens|crescent|market|arcade|parade|approach|bank|bar|"
    r"boulevard|court|croft|edge|footway|gate|green|hill|island|junction|maltings|"
    r"mews|mount|passage|point|promenade|spinney|square|terrace|thoroughfare|villas|wood"
)

FLAT_UNIT_PATTERN = r"apartment|flat|penthouse"
UNIT_PATTERN = r"(?:flat|apartment|unit)s?"
COMMON_ENDINGS = r"(?=,| and|$|;|:|\()"


# ============================================================================
# BUILDING LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def building_regex_fn(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(multi_word_no_land+building_regex+r"\b(?=,| and|$|;|:|\(|\s\d+,)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': BUILDING
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def building_special_fn(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(multi_word_no_land+building_special+r"\b(?=,| and|$|;|:|\()", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': BUILDING
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def company_name(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(r"[a-z\(\)'\s&]+"+company_type_regex, flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': COMPANY
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def mill(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    mill_regex = r"(?<=[0-9]|,|\))\s([a-z][a-z'\s]+mill)"
    end = r"(?=,| and|$|;|:)"
    search_pattern = re.compile(mill_regex+end, flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(1)[0],
            'end': m.span(1)[1],
            'label': BUILDING
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def number_comma_building_fn(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(r"(?:flat|apartment|unit)s? (?:[a-z]|\d+[a-z0-9\-\.]*|"+xx_to_yy_regex+"),\s([a-z\s.']+)(?=,)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(1)[0],
            'end': m.span(1)[1],
            'label': BUILDING
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def start_only_letters_fn(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(starting_building+r"\b(?=,| and|$|;|:|\()", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': BUILDING
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def wharf_building(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(r"(([a-z']+\s)+(wharf|quay|croft|priory|maltings|locks|gardens))(?:,)(?=[0-9a-z'\s]+"+road_regex+")", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(1)[0],
            'end': m.span(1)[1],
            'label': BUILDING
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def word_building_number(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(r"(block|building) ([a-z]([0-9]?)|\d+|)(?=,)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': BUILDING
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# CITY LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def city_match2(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(r".*,\s*([^,]*?)(?=(?:(\sand\s|\s?\(?"+postcode_regex+r")[^,]*)?$)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(1)[0],
            'end': m.span(1)[1],
            'label': CITY
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# NUMBER FILTER LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def number_filter(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(r"(?<=\()(odd|even)s?", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': NUMBER_FILTER
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# POSTCODE LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def postcodes(x: DataPoint) -> Union[List[Dict], int]:
    """Mark all matches of a regular expression within the document text as a span."""
    search_pattern = re.compile(postcode_regex, flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': POSTCODE
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# STREET NAME LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def knightsbridge_road(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(?<=\d\s)knightsbridge(?=,)")
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def meadows_regex(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(?<=[0-9]\s)[a-z][a-z'\s]*meadow(s)?(?=,| and|$|;|:)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def number_space_multi_words_roadtitle_fn(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(?<=\d\s)(\b[a-z]+\s){2,5}("+road_regex+")")
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def park_roads(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"\b[a-z][\sa-z']*park"+r"(?=,| and| \(|$)", flags=re.IGNORECASE)
    
    if getattr(x, 'commercial_park_tag', False):
        return ABSTAIN
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def road_followed_city(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(multi_word_no_land + road_regex +r"(?=\s"+city_regex+")", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def road_names_basic(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(multi_word_no_land+road_regex+r"(?=,| and| \(|$|;|:|\s"+city_regex+")", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def side(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(new road|lodge|carr moor|lake|chase|thames|kennet|coppice|church|pool) side", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def special_street_names(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(special_streets + r"(?=,| and| \(|$)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def special_welsh(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(welsh_streets, flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def the_dales(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(kirk|rye|avon|willow|darely |glen|thrush|nidder|moorside |arkengarth|wester|deep|fern|grise|common|moss)dale(?=,| and|$)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def welsh_pattern(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"\b(lon|llys|fford|clos)\s[a-z\-\s']+(?=,| and|$)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def wharf_road(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"("+multi_word_no_land+r"\b(wharf|quay(s)?|approach|parade|field(s)?|croft|priory|maltings|locks|gardens))(?:,)(?![0-9a-z'\s]+"+road_regex+")", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(1)[0],
            'end': m.span(1)[1],
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def words_waygate_fn(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(multi_word_no_land+waygate_regex+r"(?=,| and| \()")
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# STREET NUMBER LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def adjective_space_number_words_roadtitle_fn(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(?<=\b)([a-z]+\s){1,3}(\d+)(\s[a-z]+)?(\sroad|\sstreet|\savenue|\sclose|\sdrive|\splace|\sgrove|\sway|\sview|\slane|\srow|\swharf|\squay|\smeadow|\sfield|\spark|\sgardens|\screscent|\smarket|\sarcade|\sparade|\sapproach|\sbank|\sbar|\sboulevard|\scourt|\scroft|\sedge|\sfootway|\sgate|\sgreen|\shill|\sisland|\sjunction|\smaltings|\smews|\smount|\spassage|\spoint|\spromenade|\sspinney|\ssquare|\sterrace|\sthoroughfare|\svillas|\swood)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET_NUMBER
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def and_space_n(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"and\s(\d+)")
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(1),
            'end': m.end(1),
            'label': STREET_NUMBER
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def begins_with_number(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"^(\d+)")
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(1),
            'end': m.end(1),
            'label': STREET_NUMBER
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def comma_space_number_words_roadtitle_fn(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r",\s(\d+)(\s[a-z]+){1,3}(\sroad|\sstreet|\savenue|\sclose|\sdrive|\splace|\sgrove|\sway|\sview|\slane|\srow|\swharf|\squay|\smeadow|\sfield|\spark|\sgardens|\screscent|\smarket|\sarcade|\sparade|\sapproach|\sbank|\sbar|\sboulevard|\scourt|\scroft|\sedge|\sfootway|\sgate|\sgreen|\shill|\sisland|\sjunction|\smaltings|\smews|\smount|\spassage|\spoint|\spromenade|\sspinney|\ssquare|\sterrace|\sthoroughfare|\svillas|\swood)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(1),
            'end': m.end(1),
            'label': STREET_NUMBER
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def no_road_near(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"\b(no\sroad\snear)\b", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET_NUMBER
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def xx_to_yy(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(xx_to_yy_regex)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': STREET_NUMBER
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# UNIT ID LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def and_space_n_flats(x: DataPoint) -> Union[List[Dict], int]:
    if not getattr(x, 'flat_tag', False):
        return ABSTAIN
    
    search_pattern = re.compile(r"(?<=and\s)(\d+(\w{1})?)")
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def begins_with_number_flat(x: DataPoint) -> Union[List[Dict], int]:
    if not getattr(x, 'flat_tag', False):
        return ABSTAIN
    
    search_pattern = re.compile(r"^(\d+\w?)")
    match = search_pattern.search(x.text)
    
    if match:
        return [{
            'start': match.start(),
            'end': match.end(),
            'label': UNIT_ID
        }]
    
    return ABSTAIN


@labeling_function()
def carpark_id_fn(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"^(the )?(garage(s)?( space)?|parking(\s)?space|parking space(s)?|car park(ing)?( space))\s([a-z0-9\-\.]+\b)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(9)[0],
            'end': m.span(9)[1],
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def comma_space_number_comma_flat(x: DataPoint) -> Union[List[Dict], int]:
    if not getattr(x, 'flat_tag', False):
        return ABSTAIN
    
    search_pattern = re.compile(r"(?<=,\s)(\d+([a-z])?)(?=,)")
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def flat_letter(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(apartment|flat|penthouse)\s([a-z][0-9\.\-]*)(?=,)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(2)[0],
            'end': m.span(2)[1],
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def flat_letter_complex(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(apartment|flat|penthouse)\s([a-z](\d|\.|-)[a-z0-9\-\.]*)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(2)[0],
            'end': m.span(2)[1],
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def flat_number(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"(apartment|flat|penthouse)\s(\d+[a-z0-9\-\.]*)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(2)[0],
            'end': m.span(2)[1],
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def number_space_and_flat(x: DataPoint) -> Union[List[Dict], int]:
    if not getattr(x, 'flat_tag', False):
        return ABSTAIN
    
    search_pattern = re.compile(r"(?<!(-))\s(\d+)(?=\sand)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(2)[0],
            'end': m.span(2)[1],
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def unit_id_fn(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"unit\s([a-z0-9\-\.]+)", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(1)[0],
            'end': m.span(1)[1],
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def unit_letter(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"unit\s([a-z])", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.span(1)[0],
            'end': m.span(1)[1],
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def xx_to_yy_flat(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(xx_to_yy_regex)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': UNIT_ID
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# UNIT TYPE LABELLING FUNCTIONS
# ============================================================================

@labeling_function()
def flat_unit(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"flat|apartment|penthouse", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': UNIT_TYPE
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def is_carpark(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"car\s?park", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': UNIT_TYPE
        })
    
    return spans if spans else ABSTAIN


@labeling_function()
def is_other_class(x: DataPoint) -> Union[List[Dict], int]:
    search_pattern = re.compile(r"garage|storage|store|shed|outbuilding|barn|workshop|office|studio|annexe|annex|loft|cellar|basement|utility|plant|boiler|meter|cupboard|bin|cycle|pram|buggy|cleaner|caretaker|porter|concierge|warden|lobby|entrance|hall|stair|lift|escalator|corridor|passage|landing|balcony|terrace|veranda|roof|garden|yard|drive|path|lane|track|road|street|avenue|close|drive|place|grove|way|view|lane|row|wharf|quay|meadow|field|park|gardens|crescent|market|arcade|parade|approach|bank|bar|boulevard|court|croft|edge|footway|gate|green|hill|island|junction|maltings|mews|mount|passage|point|promenade|spinney|square|terrace|thoroughfare|villas|wood", flags=re.IGNORECASE)
    
    spans = []
    for m in search_pattern.finditer(x.text):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'label': UNIT_TYPE
        })
    
    return spans if spans else ABSTAIN


# ============================================================================
# MASTER FUNCTION LIST
# ============================================================================

# Master list of all labeling functions
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