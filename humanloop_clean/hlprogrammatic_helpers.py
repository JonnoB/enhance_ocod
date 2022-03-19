###
### These are useful regex's to be used in the humanloop programmatic part of the work
### By adding them here I can reuse them easily and keep them updated in a much more straight forword way
###

#this is a key regex that allows me to identify roads
road_regex  = r"(\b([a-z'])+\s)+((road|street|lane|way|gate|avenue|close|drive|hill|place|terrace|crescent|gardens|square|walk|grove|mews|row|view))\b(east|west|north|south)?"
waygate_regex = r"\b(([a-z'])+\s)*([a-z])+(way|gate)(\b(east|west|north|south))?(?=,|\sand)"


postcode_regex = r"\b[a-z]{1,2}\d[a-z0-9]?\s\d[a-z]{2}\b"

city_regex = r"(london|birmingham|manchester|liverpool|leeds|sheffield|brighton|leicester|newcastle|southhampton|portsmouth|cardiff|coventry|swansea|reading|sunderland)"

building_regex = r"(house|inn|hotel|office|centre|building|farm|cottage|lodge|home)"

xx_to_yy_regex = r"((\d+)(\s?(to|-)\s?)(\d+)\b)"

other_classes_regex = r"(land|airspace|unit|plot|store|storage)"
businesses_regex = r"((cinema)|(hotel)|(office)|(pub)|(business)|(cafe)|(restaurant)|(unit)|(store))"
company_type_regex = r"(company|ltd|limited|plc)"
