#' calc all unconvetional property
#'
#'In order to estimate the total amount of unconventional property in the dataset
#'I need to find the joint probability of a property not being unconventional
#'1-P(conventions) = =P(unconventional) =1- sum(1-P(type x)). Where type x is
#'a given type of unconventional property for all types of unconventional property
#'in this case, airnbnb, offshore, low-use
#'
#' @param data_df a dataframe the all_variables dataframe, has columns
#' 'homes','low_use', 'airbnb', 'offshore' the values of which are positive
#' integers representing the total number of each property in that geographic region
#' @param return_enhanced_df logical. Whether the unconvetional counts are
#' added to the original dataframe or a different dataframe is returned containing
#' additional data on the calculation. Defaults to TRUE
#' 
#' 
#' @returns one of two dataframes the first case the original dataframe is returned
#' with two new columns. A column of the counts of unconventional property using the
#' null hypothesis that there is no relationship between unconventional property types
#' and unconventional_overlapped which assumes that all offshore properties are
#' low_use.
#' 
#' @export

calc_all_unconventioal = function(data_df, return_enhanced_df = TRUE){
  #there is one lsoa uin newham where there are more offshore properties than homes.
  #this is probably because the ons data has not yet been updated
  #To deal with the fraction is created by taking the largest of homes or offshore
  total_unconventional_housing <- data_df[,c('homes','low_use', 'airbnb', 'offshore')] %>% 
    ungroup %>%
    mutate(across(.cols = -homes, .fns = ~{.x/pmax(homes, offshore)}),
           conventional_frac = (1-airbnb)*(1-low_use)*(1-offshore),
           unconventional_frac = 1-conventional_frac,
           conventional_overlapped_frac = (1-airbnb)*(1-pmax(low_use, offshore)),
           #If we assume that all offshore domestic properties are low-use
           #then the total number of unconventioal properties drops
           unconventional_overlapped_frac = 1- conventional_overlapped_frac,
           diff = conventional_frac-conventional_overlapped_frac,
           unconventional = round(unconventional_frac * pmax(homes, offshore)),
           unconventional_overlapped = round(unconventional_overlapped_frac * pmax(homes, offshore))) 
  
  if( return_enhanced_df){
    out = data_df %>%
      bind_cols(total_unconventional_housing %>% select(unconventional, unconventional_overlapped))
    
  } else{
    
    out <- total_unconventional_housing
  }
  
  return(out)
  
}