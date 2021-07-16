#' Monte carlo statified dataset
#' 
#' The function calls aggregated_monte_carlo_dt for each borough in London
#' 
#' 
#' @param df A data frame containing the quantities to be samples
#' @param variables a string vector. The name of the variables that provide the counts of the exclusive groups to be sampled
#' @param prices the prices for each of the sub geographies to be samples
#' @param size the total number of samples to do
#' 
#' @details this is a convenience function 
#' 
#' @return a list containing two dataframes. a data frame of all the samples for each of the local authorities, a data frame of the time
#' taken for the sampling.
#' 
#' @export
#' 
monte_carlo_stratified_dataset <- function(df, variables, prices, size){
  
  out <- unique(df$LAD11CD) %>%
    map(~{
      
      df <- df %>% 
        filter(LAD11CD ==.x #, !is.na(Homes)
               )  %>%
        #mutate(occupied = Homes- LowUse) %>%
        #select(LSOA11CD, low_use = LowUse, occupied, LAD11CD) %>%
        pivot_longer(., cols = variables, names_to = "class", values_to = "counts") %>%
        mutate(class_code = as.factor(class) %>% as.integer)
      
      prices2 <- prices %>% filter(LAD11CD == .x ) %>%
        select(sales_price = X2, LSOA11CD)
      
      start_time <- Sys.time()
      sample_df <- aggregated_monte_carlo_dt( df, prices2, number_instances = size) %>%
        mutate(LAD11CD = .x)
      end_time <- Sys.time()
      
      
      print(end_time-start_time)
      
      return(list(samples = sample_df, time = tibble(LAD11CD = .x, time_taken = difftime(end_time, start_time,, units = "secs"))))
      
    }) %>% purrr::transpose() %>% 
    map(~bind_rows(.x))
  
  
}
