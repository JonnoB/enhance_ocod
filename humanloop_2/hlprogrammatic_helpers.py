###
### These are useful regex's to be used in the humanloop programmatic part of the work
### By adding them here I can reuse them easily and keep them updated in a much more straight forword way
###

#this is a key regex that allows me to identify roads
road_regex  = r"((road|street|lane|way|gate|avenue|close|drive|hill|place|terrace|crescent|square|walk|grove|mews|row|view|boulevard|pleasant|vale|yard|chase|rise|green|passage|friars|viaduct|promenade|\bend|\bridge|embankment|villas|circus|\bpath|pavement))\b( east| west| north| south)?"

multi_word_no_land = r"(?<!\S)(?:(?!\b(?:at|on|in|adjoining|of|off|and|to)\b)[^\),\n\d])*?\s?"
#based on the above but with lots of carveouts
starting_building  = r"^(?<!\S)(?:(?!\b(?:apartment\b|penthouse|flat|ground|basement|suite|\broom|(first|second|third|fourth|fith|sixth|seventh) floor|(the )?(airspace|land|plot|unit|car|parking|store|storage)|at|on|in|adjoining|of|off|and|to)\b)[^\),\n\d])*?\s?"
waygate_regex = r"(way|gate)(\b(east|west|north|south))?"

welsh_streets = r"(pendre|ryw blodyn|bryn owain|pen y dre|maes yr haf|heol staughton|glantraeth|tai maes|hafod alyn|cae alaw goch|ynys y wern|dol isaf|bro deg|eglwys teg|heol-y-frenhines|downleaze cockett|waun daniel|twyni teg|llwyn onn|delffordd|coed-y-brain|waun Y felin|glan Y lli|ty-draw)"


special_streets = r"((kensington gore|wilds rents|the mound|high holborn|pall mall|lower mall|haymarket|lower marsh|marsh wall|whyke marsh|london wall|cheapside|eastcheap|piccadilly|aldwych|the strand|strand|bevis marks|old bailey|threelands|pendenza|castelnau|the old meadow|hortonwood|thoroughfare|navigation loop|turnberry|brentwood|hatton garden|greenacres|whitehall|the quadrangle|green lanes|old jewry|st mary axe|minories|foxcover|meadow brook|daisy brook|upper ground|march wall|millharbour|aztec west|trotwood|marlowes|petty france|petty cury|the quadrant|the spinney|robins corner|houndsditch|frogmoor|hanging birches|the birches|arthurstone birches|monks wood|the cedars|the meadows|sandiacre|millbank|moorfields))"

postcode_regex = r"\b[a-z]{1,2}\d[a-z0-9]?\s\d[a-z]{2}\b"


city_regex = r"(london|birmingham|manchester|liverpool|leeds|sheffield|brighton|leicester|newcastle|southhampton|portsmouth|cardiff|coventry|swansea|reading|sunderland)"

building_regex = r"\b(school|church|workshops|court|house|inn|tavern|hotel|cinema(s)?|office|centre|center|building(s)?|bungalow|[a-z]*works|farm|cottage|lodge|home|point|arcade(s)?|institute|hall|mansions|country club|apartments( east| south| west| north)?|(tower(s)?)(\s\w+)?)"

xx_to_yy_regex = r"((\d+)(\s?(to|-)\s?)(\d+)\b)"

building_special = r"(the knightsbridge(?=,)|chichester rents|belgravia gate|the chilterns|the Belvedere|(one hyde park)(?=,)|park plaza westminster bridge|"+multi_word_no_land+"exchange)"

other_classes_regex = r"(airspace|unit|plot|store|storage|storage pod|\broom|suite|studio)"
businesses_regex = r"((cinema)|(hotel)|(office)|(pub)|(business)|(cafe)|(restaurant)|(unit)|(store))"
company_type_regex = r"(company|ltd|limited|plc)"