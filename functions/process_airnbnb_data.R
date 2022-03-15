#' Process airbnb data
#' 
#' 
#' This is a convencience function ro process the aribnb data into a usable form for the purposes of this project
#' 
#'  @param LSOAshapedata file path of the shape files of the lsoa data
#'  @param CorePstCd dataframe of the postcode look up to geographies dictionary
#'  @param airbnb_csv path to the airbnb data
#' 
#'  @details The function currently only includes properties if they are the entire home.
#' 
#'  @return A dataframe containing two columns, the LSOA code and the number of airbnb properties per lsoa. 
#' 
#' @export


process_airnbnb_data <- function(LSOAshapedata, CorePstCd, airbnb_csv ){
  
  
  LSOAshape <- st_read(LSOAshapedata) %>%
    #Use the postcode lookup to map LSOA and region to the dataset. Filter by region == LONDON
    left_join(CorePstCd %>%
                distinct(LSOA11CD, .keep_all = TRUE) %>%
                select(-Postcode), by = c("lsoa11cd"="LSOA11CD")) %>%
   # filter(Region %in% "E12000007") %>%
    st_transform(., crs = 4326) %>%
    st_make_valid() #some of the shapes overlap are incomplete this ensures they close properly
  
  airbnb_df <- read_csv(airbnb_csv) %>%
    filter(room_type == "Entire home/apt") %>%
    st_as_sf(., coords = c("longitude", "latitude"), crs = st_crs(LSOAshape)) %>%
    mutate(intersection = as.integer(st_intersects(geometry, LSOAshape)),
           LSOA11CD = LSOAshape$lsoa11cd[intersection]) %>%
    #a few fall outside the London LSOA. These are removed
    filter(!is.na(intersection))
  
  airbnb_lsoa_counts <- airbnb_df %>%
    group_by(LSOA11CD) %>%
    summarise(airbnb = n()) %>%
    as_tibble(.) %>%
    select(-geometry)
  
  return(airbnb_lsoa_counts)
  
}