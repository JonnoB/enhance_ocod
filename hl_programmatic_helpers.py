###
### These are useful regex's to be used in the humanloop programmatic part of the work
### By adding them here I can reuse them easily and keep them updated in a much more straight forword way
###

#this is a key regex that allows me to identify roads
road_regex  = r"(?<!\S)(?:(?!\b(?:at|on|in|to|of|adjoining)\b)[^\n\d])*? ((road|street|lane|way|gate|avenue|close|drive|hill|place|terrace|crescent|gardens|square|walk|grove|mews|row|quay))\b( east| west| north| south)?"

