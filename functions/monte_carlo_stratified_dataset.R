#' Monte carlo statified dataset
#' 
#' The function calls aggregated_monte_carlo_dt for each borough in London
#' 
#' 
#' @param df A data frame containing the quantities to be samples
#' @param variables a string vector. The name of the variables that provide the counts of the exclusive groups to be sampled
#' @param prices the prices for each of the sub geographies to be samples
#' @param size the total number of samples to do
#' @param geography_name a chracter string. The name of the variable that contains the geography IDs, typically LSOA11CD
#' 
#' @details this is a convenience function 
#' 
#' @return a list containing two dataframes. a data frame of all the samples for each of the local authorities, a data frame of the time
#' taken for the sampling.
#' 
#' @export
#' 
monte_carlo_stratified_dataset <- function(df, variables, prices, size, geography_name = "LSOA11CD"){
  
  out <- unique(df$LAD11CD) %>%
    map(~{
      #print(.x)
      df <- df %>% 
        filter(LAD11CD ==.x )  %>%
        pivot_longer(., cols = variables, names_to = "class", values_to = "counts") %>%
        mutate(class_code = as.factor(class) %>% as.integer) %>%
        #aggregate to make sure that the data has one entry per geography_name and class
        group_by(.data[[geography_name]], class, class_code) %>%
        summarise(counts = sum(counts)) %>%
        ungroup
      
      prices2 <- prices %>% filter(LAD11CD == .x ) %>%
        select(sales_price = X2, geography_name)
      
      start_time <- Sys.time()
      sample_df <- aggregated_monte_carlo_dt( df2 =df, prices2, number_instances = size, geography_name = geography_name) %>%
        mutate(LAD11CD = .x)
      end_time <- Sys.time()
      
      
      print(end_time-start_time)
      
      return(list(samples = sample_df, time = tibble(LAD11CD = .x, time_taken = difftime(end_time, start_time,, units = "secs"))))
      
    }) %>% purrr::transpose() %>% 
    map(~bind_rows(.x))
  
  
}
