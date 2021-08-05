#' Create Sampled vector
#'
#' An internal function that creates a very long vector representing the entire monte-carlo simulation
#' 
#'  @param geography_counts A data frame. has two columns smallest_unit which the ID code of the geography and 
#'  the total number in that geography
#'  @param prices2 A data frame. has two columns sales_price and LSOA11CD the ID code of the LSOA
#'
#'  @details not a lot of detail
#'
#'  @return returns a vector that is the stratified sample across all units for each instance of the monte-carlo simulation
#'
#'  @export
#'
create_sampled_vector <- function(geography_counts, prices2, samples = 501, geography_name){
  
  sample_vect <- vector(mode = "list", length = nrow(geography_counts))
  
  geography_ids <- unique(geography_counts$smallest_unit)
  
  for(q in 1:length(geography_ids)){
    
    target_unit <- geography_ids[q]
    
    unit_values <- prices2$sales_price[prices2[[geography_name]]==target_unit] 
    
    #Sometimes an LSOA/MSOA will have no sales in that year. In these thankfully rare cases the entire LAD is used.
    if(length(unit_values)==0){
      unit_values <- prices2$sales_price
    }
    
    counts <- geography_counts$total[geography_counts$smallest_unit == target_unit]
    
    sample_vect[[q]] <- sample(unit_values, 
                               size = counts*samples,
                               replace = TRUE)
    
  }
  
  return(unlist(sample_vect))
  
}