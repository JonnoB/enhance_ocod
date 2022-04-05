#'
#' aggregate unit values NO LONGER NEEDDED< REPLACED BY DATATABLE
#' 
#'
#' An internal function that aggregates the sampled vector to the higherlevel
#'
#' @param presized_list a list of n numeric vectors of length number_of instances. n is the number of
#' mutually exclusive classes plus 1 for the total
#' @param sampled_vect a numeric vector. A the ful monte carlo simulation in vector form
#' @param instance_id_vect a numeric vector. A vector giving the monte-arlo instance ID, this is used for aggregating
#' @param unit_id_vect a numeric vector. This gives the id of the smallest unit that is being aggregated to the total level.
#' @param number_instances an integer. the total number of monte-carlo instances
#' 
#' 
#' @details This internal function is to clean up the code in aggregated_monte_carlo and help make optimisiation easier
#' 
#' @return returns the pre_sized_list with the elements filled out.
#' 
#' @export
#'
aggregate_unit_values <- function(pre_sized_list, sampled_vect, instance_id_vect, unit_id_vect, number_instances){
for(.x in 1:number_instances){
  print(paste("Aggregating instance",.x))
  #Doing the subset is the most expensive task so it should only be done as little as possible.
  instance_vect <- instance_id_vect==.x
  #subset the class vector to only this instance
  instance_class_id_vect <- unit_id_vect[instance_vect]
  #take the sample vector for only this instance
  temp <- sampled_vect[instance_vect]
  
  pre_sized_list[[1]][[.x]] <- mean(temp)
  
  #create the values for the sub-classes as well to save on the assigning
  sub_classes <-for( n in  1:nrow(class_dict)){
    #subset temp again using the pre-subset class vector, so that temp is only relevant to the active class
    temp2 <- temp[instance_class_id_vect==n]
    pre_sized_list[[class_dict$class[n]]][.x] <- mean(temp2)
    
    
  }
}
  return(pre_sized_list)
}