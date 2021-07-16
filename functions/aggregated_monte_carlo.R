#' Aggregated monte-carlo
#' 
#' Creates a dataframe of summarised monte-carlo simulations aggregated from base unit level to higher level for n
#' mutually exclusive groups
#' 
#'  @param df2 a dataframe inlcudes the columns LSOA11CD, class and counts
#'  @param prices2 A data frame. has two columns sales_price and LSOA11CD the ID code of the LSOA
#'  @param number_of_instances A numeric. the total number of monte-carlo instances to perform
#'
#'  @details This is really useful and quite fast.
#'  
#'  @export
#'
aggregated_monte_carlo_dt <- function(df2, prices2, number_instances = 501){
  
  df2 <- df2 %>%
    mutate(class_code = as.factor(class) %>% as.integer(.))
  
  sample_counts <- df2 %>%
    rename(smallest_unit = LSOA11CD) %>%
    group_by(smallest_unit) %>%
    summarise(total = sum(counts)) 
  
  number_units <- nrow(sample_counts)
  #this produces a vector of the sampl1e ID's where each element represents one instance of one unit in the dataset
  #therefore for a dataset of 5 units and 10 instances, there will be the numbers 1:10 a total of 5 times
  sample_id_vect <- rep(1:number_instances, times = number_units)
  
  #create a vector where each element is the number of samples in each unit repeated for the total number of instances
  #that will be generated
  sample_instances_vect <- rep(sample_counts$total, each = number_instances)
  
  #each unique value in this vector identifies one instance
  instance_id_vect <- rep(sample_id_vect, times = sample_instances_vect)
  
  #small example of the above
  #rep(rep(1:5, times = 2), times = rep(c(2, 3), each = 5))
  
  unit_id <-unique(df2$LSOA11CD)
  
  #this is a bit of an ugly way to generate a vector where each element represents the class.
  #the classes repeat every n elements where n is the number of samples of all classes in that unit. 
  #for each instance the numbers of each class are repeated m times where m is the number of observations of that
  #class in that unit.
  #It may be ugly but it is not very slow so I don't care
  unit_id_vect <- 1:length(unit_id) %>%
    map(~{
      per_unit_df <- df2[df2$LSOA11CD== unit_id[.x],]
      #generate the class code vectors
      out <- 1:nrow(per_unit_df) %>% map(~rep(per_unit_df$class_code[.x], each = per_unit_df$counts[.x])) %>% unlist %>%
        rep(., times = number_instances)
      
    }) %>% unlist
  
  print("Creating Monte-Carlo simulations")
  sampled_vect <- create_sampled_vector(sample_counts, prices2, samples = number_instances )
  
  class_dict <- df2 %>% select(class, class_code) %>% distinct()
  
  print("summarising monte-carlo")
  
  #These are very large aggregations, so data.table is used
  data_to_summarise_df <- data.table(sampled_vect, instance_id_vect, unit_id_vect)
  
  #first create the averages for all groups
  total_summary_df <- data_to_summarise_df[, mean(sampled_vect), keyby = instance_id_vect][,"V1"] %>%
    as_tibble(.) %>%
    rename(total = V1)
  
  #then create the averages for the sub-groups
  out <-1:nrow(class_dict) %>%
    map(~{
      #find the mean for each instance for each of the sub-groups
      data_to_summarise_df[unit_id_vect == .x][, mean(sampled_vect), keyby = instance_id_vect][,"V1"] %>%
        as_tibble(.) %>%
        rename(!!sym(class_dict$class[.x]) :=V1)
      
    }) %>% 
    #glue it all together and rename ready for output
    bind_cols() %>%
    bind_cols(total_summary_df, .) %>%
    mutate(id = 1:n()) %>%
    select(id, everything())
  
  return(out)
  
}
