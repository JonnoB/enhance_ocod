"""
Regular expressions for Named Entity Recognition (NER) labelling in property address data.

This module contains both raw regex pattern strings and their pre-compiled versions
for improved performance. Pattern strings are maintained for documentation and reuse,
while compiled versions are used in labelling functions.
"""

import re

# ============================================================================
# RAW PATTERN STRINGS (for reference and composition)
# ============================================================================

# Road identification regex
road_regex = r"((road|street|lane|way|gate|avenue|close|drive|hill|place|terrace|crescent|square|walk|grove|mews|row|view|boulevard|pleasant|vale|yard|chase|rise|green|passage|friars|viaduct|promenade|\bend|\bridge|embankment|villas|circus|\bpath|pavement))\b( east| west| north| south)?"

multi_word_no_land = r"(?<!\S)(?:(?!\b(?:at|on|in|adjoining|of|off|above|being|and|to)\b)[^\),\n\d])*?\s?"

starting_building = r"^(?<!\S)(?:(?!\b(?:apartment\b|penthouse|flat|ground|basement|suite|\broom|(first|second|third|fourth|fith|sixth|seventh) floor|(the )?(airspace|land|plot|unit|car|parking|store|storage)|at|on|in|adjoining|of|off|and|to)\b)[^\),\n\d])*?\s?"

waygate_regex = r"(way|gate)(\b(east|west|north|south))?"

building_special = (
    r"(the knightsbridge(?=,)|lake shore(?=,)|chichester rents|20:20 house|travelodge|little chef|the forge|(?<=\s)x1 [a-z\s]+(?=,)|the cube|belgravia gate|the chilterns|the Belvedere|(one hyde park)(?=,)|park plaza westminster bridge|"
    + multi_word_no_land
    + "exchange)"
)

special_streets = r"((kensington gore|wilds rents|the mound|high holborn|pall mall|lower mall|haymarket|lower marsh|marsh wall|whyke marsh|london wall|cheapside|eastcheap|piccadilly|aldwych|the strand|strand|bevis marks|old bailey|threelands|pendenza|castelnau|the old meadow|hortonwood|thoroughfare|navigation loop|turnberry|brentwood|hatton garden|greenacres|whitehall|the quadrangle|green lanes|old jewry|st mary axe|minories|foxcover|meadow brook|daisy brook|upper ground|march wall|millharbour|aztec west|trotwood|marlowes|petty france|petty cury|the quadrant|the spinney|robins corner|houndsditch|frogmoor|hanging birches|the birches|arthurstone birches|monks wood|the cedars|the meadows|sandiacre|millbank|moorfields))"

welsh_streets = r"(pendre|ryw blodyn|bryn owain|pen y dre|maes yr haf|heol staughton|glantraeth|tai maes|hafod alyn|cae alaw goch|ynys y wern|dol isaf|bro deg|eglwys teg|heol-y-frenhines|downleaze cockett|waun daniel|twyni teg|llwyn onn|delffordd|coed-y-brain|waun Y felin|glan Y lli|ty-draw)"

postcode_regex = r"\b[a-z]{1,2}\d[a-z0-9]?\s\d[a-z]{2}\b"

city_regex = r"(london|birmingham|manchester|liverpool|leeds|sheffield|brighton|leicester|newcastle|southhampton|portsmouth|cardiff|coventry|swansea|reading|sunderland)"

building_regex = r"\b(school|church|workshops|court|house|inn|tavern|hotel|annex|cinema(s)?|office|centre|center|building(s)?|bungalow|[a-z]*works|farm|cottage|lodge|home|point|arcade(s)?|institute|hall|mansions|country club|apartments( east| south| west| north)?|(tower(s)?)(\s\w+)?)"

xx_to_yy_regex = r"\d+[a-z]?\s?(?:to|-|/)\s?\d+[a-z]?\b"

other_classes_regex = r"(airspace|unit|land|plot|store|storage|storage pod|storage locker|\broom|suite|studio)"

businesses_regex = r"((cinema)|(hotel)|(office)|(pub)|(business)|(cafe)|(restaurant)|(unit)|(store))"

company_type_regex = r"(company|ltd|limited|plc)"

# ============================================================================
# COMMON PATTERN FRAGMENTS (used by labelling functions)
# ============================================================================

ROAD_TYPES = (
    r"road|street|avenue|close|drive|place|grove|way|view|lane|row|wharf|quay|"
    r"meadow|field|park|gardens|crescent|market|arcade|parade|approach|bank|bar|"
    r"boulevard|court|croft|edge|footway|gate|green|hill|island|junction|maltings|"
    r"mews|mount|passage|point|promenade|spinney|square|terrace|thoroughfare|villas|wood"
)

UNIT_PATTERN = r"(?:flat|apartment|unit|penthouse|plot)s?"
FLAT_UNIT_PATTERN = r"apartment|flat|penthouse|unit|plot"
COMMON_ENDINGS = r"(?=,| and|$|;|:|\()"

# ============================================================================
# PRE-COMPILED PATTERNS (for performance)
# ============================================================================

class CompiledPatterns:
    """
    Container for all pre-compiled regex patterns used in labelling functions.
    
    Patterns are compiled once at module import time and reused across all
    function calls, providing 2-3x speedup compared to runtime compilation.
    """
    
    # Building patterns
    BUILDING_REGEX = re.compile(
        multi_word_no_land + building_regex + r"\b(?=,| and|$|;|:|\(|\s\d+,)",
        re.IGNORECASE
    )
    
    BUILDING_SPECIAL = re.compile(
        multi_word_no_land + building_special + COMMON_ENDINGS,
        re.IGNORECASE
    )
    
    MILL = re.compile(
        r"(?<=[0-9]|,|\))\s([a-z][a-z'\s]+mill)(?=,| and|$|;|:)",
        re.IGNORECASE
    )
    
    NUMBER_COMMA_BUILDING = re.compile(
        rf"{UNIT_PATTERN} (?:[a-z]|\d+[a-z0-9\-\.]*|{xx_to_yy_regex}),\s([a-z\s.']+)(?=,)",
        re.IGNORECASE
    )
    
    START_ONLY_LETTERS = re.compile(
        starting_building + COMMON_ENDINGS,
        re.IGNORECASE
    )
    
    WHARF_BUILDING = re.compile(
        rf"(([a-z']+\s)+(wharf|quay|croft|priory|maltings|locks|gardens))(?:,)(?=[0-9a-z'\s]+{road_regex})",
        re.IGNORECASE
    )
    
    WORD_BUILDING_NUMBER = re.compile(
        r"(block|building) ([a-z]([0-9]?)|\d+|)(?=,)",
        re.IGNORECASE
    )
    
    COMPANY_NAME = re.compile(
        r"[a-z\(\)'\s&]+" + company_type_regex,
        re.IGNORECASE
    )
    
    # City patterns
    CITY_MATCH = re.compile(
        rf".*,\s*([^,]*?)(?=(?:(\sand\s|\s?\(?{postcode_regex})[^,]*)?$)",
        re.IGNORECASE
    )
    
    # Number filter
    NUMBER_FILTER = re.compile(
        r"(?<=\()\s?(odd|even)(?=s?\b)",
        re.IGNORECASE
    )
    
    # Postcode
    POSTCODE = re.compile(postcode_regex, re.IGNORECASE)
    
    # Street name patterns
    KNIGHTSBRIDGE_ROAD = re.compile(r"(?<=\d\s)knightsbridge(?=,)")
    
    MEADOWS_REGEX = re.compile(
        r"(?<=[0-9]\s)[a-z][a-z'\s]*meadow(s)?(?=,| and|$|;|:)",
        re.IGNORECASE
    )
    
    NUMBER_SPACE_MULTI_WORDS_ROADTITLE = re.compile(
        r"(?<=\d\s)(\b[a-z]+\s){2,5}(" + road_regex + ")"
    )
    
    PARK_ROADS = re.compile(
        r"\b[a-z][\sa-z']*park" + r"(?=,| and| \(|$)",
        re.IGNORECASE
    )
    
    ROAD_FOLLOWED_CITY = re.compile(
        multi_word_no_land + road_regex + r"(?=\s" + city_regex + ")",
        re.IGNORECASE
    )
    
    ROAD_NAMES_BASIC = re.compile(
        multi_word_no_land + road_regex + r"(?=,| and| \(|$|;|:|\s" + city_regex + ")",
        re.IGNORECASE
    )
    
    SIDE = re.compile(
        r"(new road|lodge|carr moor|lake|chase|thames|kennet|coppice|church|pool) side",
        re.IGNORECASE
    )
    
    SPECIAL_STREET_NAMES = re.compile(
        special_streets + r"(?=,| and| \(|$)",
        re.IGNORECASE
    )
    
    SPECIAL_WELSH = re.compile(welsh_streets, re.IGNORECASE)
    
    THE_DALES = re.compile(
        r"(kirk|rye|avon|willow|darely |glen|thrush|nidder|moorside |arkengarth|wester|deep|fern|grise|common|moss)dale(?=,| and|$)",
        re.IGNORECASE
    )
    
    WHARF_ROAD = re.compile(
        r"(" + multi_word_no_land + 
        r"\b(wharf|quay(s)?|approach|parade|field(s)?|croft|priory|maltings|locks|gardens))(?:,)(?![0-9a-z'\s]+" +
        road_regex + ")",
        re.IGNORECASE
    )
    
    WORDS_WAYGATE = re.compile(
        multi_word_no_land + waygate_regex + r"(?=,| and| \()"
    )
    
    # Street number patterns
    ADJECTIVE_SPACE_NUMBER_WORDS_ROADTITLE = re.compile(
        r"(?:\b(?:at|on|in|adjoining|of|above|off|and|being|next to)\s)(\d+[a-z]?)(?=" +
        r"(\s([a-z])+('s)?)+(\s" + road_regex + r")?" + r")"
    )
    
    AND_SPACE_N = re.compile(r"(?<=and\s)(\d+[a-z]?)\b")
    
    BEGINS_WITH_NUMBER = re.compile(r"^(\d+([a-z])?)\b(?!(\s?(to|-)\s?)(\d+)\b)")
    
    COMMA_SPACE_NUMBER_WORDS_ROADTITLE = re.compile(
        r"(?<=,\s)\d+[a-z]?(?=" + r"(\s([a-z])+('s)?)+(\s" + road_regex + r")?" + r")"
    )
    
    NO_ROAD_NEAR = re.compile(
        r"(at|on|in|adjoining|of|off|and|to|being|,)\s(\d+[a-z]?)\b(?=,| and|;|:|\s\()",
        re.IGNORECASE
    )
    
    XX_TO_YY = re.compile(xx_to_yy_regex, re.IGNORECASE)
    
    XX_TO_YY_WITH_ROAD = re.compile(
        xx_to_yy_regex + r"(?=(([a-z'\s]+" + road_regex + r")|" + special_streets + r"))",
        re.IGNORECASE
    )
    
    # Unit ID patterns
    AND_SPACE_N_FLATS = re.compile(r"(?<=and\s)(\d+(\w{1})?)")
    
    BEGINS_WITH_NUMBER_FLAT = re.compile(r"^(\d+\w?)")
    
    NUMBER_BEFORE_BUILDING = re.compile(
        r"^([a-z]?\d+[a-z0-9\-\.]*),?\s+[^,]*?" + building_regex,
        re.IGNORECASE
    )
    
    CARPARK_ID = re.compile(
        r"^(the )?(garage(s)?( space)?|parking(\s)?space|parking space(s)?|car park(ing)?( space))\s([a-z0-9\-\.]+\b)",
        re.IGNORECASE
    )
    
    COMMA_SPACE_NUMBER_COMMA_FLAT = re.compile(r"(?<=,\s)(\d+([a-z])?)(?=,)")
    
    FLAT_LETTER = re.compile(
        UNIT_PATTERN + r"\s([a-z][0-9\.\-]*)(?=,)",
        re.IGNORECASE
    )
    
    FLAT_LETTER_COMPLEX = re.compile(
        UNIT_PATTERN + r"\s([a-z](\d|\.|-)[a-z0-9\-\.]*)",
        re.IGNORECASE
    )
    
    FLAT_NUMBER = re.compile(
        UNIT_PATTERN + r"\s(\d+[a-z0-9\-\.]*)",
        re.IGNORECASE
    )
    
    NUMBER_SPACE_AND_FLAT = re.compile(
        r"(?<!(-))\s(\d+)(?=\sand)",
        re.IGNORECASE
    )
    
    UNIT_ID = re.compile(
        r"(the )?" + other_classes_regex + r"(s)?\s(\d[a-z0-9\-\.]*)",
        re.IGNORECASE
    )
    
    UNIT_LETTER = re.compile(
        other_classes_regex + r"s?\s([a-z](?=,| and)|[a-z](\d|\.|-)[a-z0-9\-\.]*)",
        re.IGNORECASE
    )
    
    XX_TO_YY_FLAT = re.compile(
        xx_to_yy_regex + r"(?!\s(being|-|[a-z'\s]+" + road_regex + r"|" + special_streets + "))",
        re.IGNORECASE
    )
    
    # Unit type patterns
    FLAT_UNIT = re.compile(
        r"(apartment|flat|penthouse|unit|plot)\b(?!being)",
        re.IGNORECASE
    )
    
    IS_CARPARK = re.compile(
        r"^(garage(s)?|parking(\s)?space|parking space(s)?|car park(ing)?)",
        re.IGNORECASE
    )
    
    IS_OTHER_CLASS = re.compile(other_classes_regex, re.IGNORECASE)


# Create instance for convenient access
PATTERNS = CompiledPatterns()

# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

# Export both raw strings (for documentation/composition) and compiled patterns
__all__ = [
    # Raw pattern strings
    'road_regex',
    'multi_word_no_land',
    'starting_building',
    'waygate_regex',
    'building_special',
    'special_streets',
    'welsh_streets',
    'postcode_regex',
    'city_regex',
    'building_regex',
    'xx_to_yy_regex',
    'other_classes_regex',
    'businesses_regex',
    'company_type_regex',
    # Compiled patterns
    'PATTERNS',
    'CompiledPatterns',
]