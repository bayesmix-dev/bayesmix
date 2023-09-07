#' Run BayesMix MCMC samplers
#'
#' In this light version, this function calls the BayesMix executable from a subprocess via \code{\link[base]{system}} command.
#'
#' @param hierarchy A string, the id of the hierarchy. Must be one of the 'Name' in the [\code{hierarchy_id.proto}](http://bayesmix.readthedocs.io/en/latest/protos.html#hierarchy_id.proto) list.
#' @param mixing A string, the id of the mixing. Must be one of the 'Name' in the [\code{mixing_id.proto}](http://bayesmix.readthedocs.io/en/latest/protos.html#mixing_id.proto) list.
#' @param data A numeric vector or matrix of shape (n_samples, n_dim). These are the observations on which to fit the model.
#' @param hier_params A text string containing the hyperparameters of the hierarchy or a file name where the hyperparameters are stored.
#' A protobuf message of the corresponding type will be created and populated with the parameters.
#' See the file [\code{hierarchy_prior.proto}](https://bayesmix.readthedocs.io/en/latest/protos.html#hierarchy_prior.proto) for the corresponding message.
#' @param mix_params A text string containing the hyperparameters of the mixing or a file name where the hyperparameters are stored.
#' A protobuf message of the corresponding type will be created and populated with the parameters.
#' See the file [\code{mixing_prior.proto}](https://bayesmix.readthedocs.io/en/latest/protos.html#mixing_prior.proto) for the corresponding message.
#' @param algo_params A text string containing the hyperparameters of the algorithm or a file name where the hyperparameters are stored.
#' See the file [\code{algorithm_params.proto}](https://bayesmix.readthedocs.io/en/latest/protos.html#algorithm_params.proto) for the corresponding message.
#' @param dens_grid A numeric vector or matrix of shape (n_dens_grid_points, n_dim). The are the points where to evaluate the density.
#' If \code{NULL}, the density will not be evaluated.
#' @param out_dir A string. If not \code{NULL}, is the folder where to store the output. If \code{NULL}, a temporary directory will be created and destroyed
#' after the sampling is finished.
#' @param return_clusters A bool. If \code{TRUE}, returns the chain of the cluster allocations.
#' @param return_best_clus A bool. If \code{TRUE}, returns the best cluster allocation obtained by minimizing the Binder loss function over the visited partitions during the MCMC sampling.
#' @param return_num_clusters A bool. If \code{TRUE}, returns the chain of the number of clusters.
#'
#' @return A list with the following components:
#' \itemize{
#'   \item{\strong{eval_dens} a matrix of shape (n_samples, n_dens_grid_points). It is the mixture density evaluated at the points in dens_grid for each iteration. \code{NULL} if \code{eval_dens} is \code{NULL}.}
#'   \item{\strong{n_clus} numeric vector of shape (n_samples). The number of clusters for each iteration. \code{NULL} if \code{return_num_clusters} is \code{FALSE}.}
#'   \item{\strong{clus} numeric matrix of shape (n_samples, n_data). The cluster allocation for each iteration. \code{NULL} if \code{return_clusters} is \code{FALSE}.}
#'   \item{\strong{best_clus} numeric vector of shape (n_data). The best clustering obtained by minimizing Binder's loss function. \code{NULL} if \code{return_best_clus} is \code{FALSE}.}
#' }
#'
#' @export
run_mcmc <- function(hierarchy, mixing, data,
                     hier_params, mix_params, algo_params, dens_grid = NULL, out_dir = NULL,
                     return_clusters = TRUE, return_best_clus = TRUE, return_num_clusters = TRUE) {

  # Check input types
  if (!is.character(hierarchy)) { stop("'hierarchy' parameter must be a string") }
  if (!is.character(mixing)) { stop("'mixing' parameter must be a string") }
  if (!is.numeric(data)) { stop("'data' parameter must be a numeric vector or matrix") }
  if (!is.character(hier_params)) { stop("'hier_params' parameter must be a string") }
  if (!is.character(mix_params)) { stop("'mix_params' parameter must be a string") }
  if (!is.character(algo_params)) { stop("'algo_params' parameter must be a string") }
  if (!is.numeric(dens_grid) & !is.null(dens_grid)) { stop("'dens_grid' parameter can be a numeric vector or matrix, NULL otherwise") }
  if (!is.character(out_dir) & !is.null(out_dir)) { stop("'out_dir' parameter must be a string, NULL otherwise") }
  if (!is.logical(return_clusters)) { stop("'return_clusters' parameter must be a bool") }
  if (!is.logical(return_best_clus)) { stop("'return_best_clus' parameter must be a bool") }
  if (!is.logical(return_num_clusters)) { stop("'return_num_clusters' parameter must be a bool") }

  # Get BAYESMIX_EXE and check if set
  BAYESMIX_EXE = Sys.getenv("BAYESMIX_EXE")
  if(BAYESMIX_EXE == ""){
    stop("BAYESMIX_EXE environment variable not set")
  }

  # Set-up template for run_mcmc command
  params = paste('--algo-params-file "%s" --hier-type "%s"',
                 '--hier-args "%s" --mix-type "%s"',
                 '--mix-args "%s" --coll-name "%s"',
                 '--data-file "%s" --grid-file "%s"',
                 '--dens-file "%s" --n-cl-file "%s"',
                 '--clus-file "%s" --best-clus-file "%s"')

  # Set run_mcmc command template
  RUN_CMD = paste(BAYESMIX_EXE, params)

  # Use temporary directory if out_dir is not set
  if(is.null(out_dir)) {
    out_dir = sprintf("%s/temp", dirname(BAYESMIX_EXE)); dir.create(out_dir)
    remove_out_dir = TRUE
  } else {
    remove_out_dir = FALSE
  }

  # Prepare files for data and outcomes
  data_file = paste0(out_dir,"/data.csv"); file.create(data_file)
  dens_grid_file = paste0(out_dir, '/dens_grid.csv'); file.create(dens_grid_file)
  n_clus_file = paste0(out_dir, '/n_clus.csv'); file.create(n_clus_file)
  clus_file = paste0(out_dir, '/clus.csv'); file.create(clus_file)
  best_clus_file = paste0(out_dir, '/best_clus.csv'); file.create(best_clus_file)
  eval_dens_file = paste0(out_dir, "/eval_dens.csv"); file.create(eval_dens_file)

  # Prepare protobuf configuration files
  hier_params_file = bayesmixr:::maybe_print_to_file(hier_params, "hier_params", out_dir)
  mix_params_file = bayesmixr:::maybe_print_to_file(mix_params, "mix_params", out_dir)
  algo_params_file = bayesmixr:::maybe_print_to_file(algo_params, "algo_params", out_dir)

  # Set-up NULL filenames for arg-parse
  utils::write.table(data, file = data_file, sep = ",", col.names = F, row.names = F)
  if(is.null(dens_grid)) {
    dens_grid_file = '""""' #"\"\""
    eval_dens_file = '""""' #"\"\""
  } else {
    utils::write.table(dens_grid, file = dens_grid_file, sep = ",", row.names = F, col.names = F)
  }
  if(!return_clusters) {
    clus_file = '""""' #"\"\""
  }
  if(!return_num_clusters) {
    nclus_file = '""""' #"\"\""
  }
  if(!return_best_clus) {
    best_clus_file = '""""' #"\"\""
  }

  # Resolve run_mcmc command
  CMD = sprintf(RUN_CMD, algo_params_file, hierarchy,
                hier_params_file, mixing,
                mix_params_file, 'memory',
                data_file, dens_grid_file,
                eval_dens_file, n_clus_file,
                clus_file, best_clus_file)
  
  # Add TBB_PATH to PATH (required in Windows)
  EXT_PATH = withr::with_path(Sys.getenv("TBB_PATH"), Sys.getenv("PATH"))

  # Execute run_mcmc in extended environment
  tryCatch({
    withr::with_path(EXT_PATH, system(CMD))
  },
  error = function(e) {
    message(sprintf("Failed with error: %s\n)", as.character(e)))
   if(remove_out_dir) {
     unlink(out_dir, recursive = TRUE)
     #unlink(paste0(out_dir,"/*.csv"))
     #unlink(paste0(out_dir,"/*.asciipb"))
   }
   return(NULL)
  })

  # Manage return arguments
  eval_dens = NULL
  if(!is.null(dens_grid)) {
      eval_dens = suppressWarnings(as.matrix(utils::read.table(eval_dens_file, sep = ",")))
      attributes(eval_dens)$dimnames <- NULL
  }
  nclus = NULL
  if(return_num_clusters){
    nclus = suppressWarnings(as.matrix(utils::read.table(n_clus_file, sep = ",")))
    attributes(nclus)$dimnames <- NULL
  }
  clus = NULL
  if(return_clusters){
    clus = suppressWarnings(as.matrix(utils::read.table(clus_file, sep = ",")))
    attributes(clus)$dimnames = NULL
  }
  best_clus = NULL
  if(return_best_clus) {
    best_clus = suppressWarnings(as.matrix(utils::read.table(best_clus_file, sep = ",")))
    attributes(best_clus)$dimnames = NULL
  }

  # Define out list
  out = list("eval_dens" = eval_dens,
             "n_clus" = as.vector(nclus),
             "clus" = clus,
             "best_clus" = as.vector(best_clus))

  # Clean temporary files and return
  if(remove_out_dir){
    unlink(out_dir, recursive = TRUE)
    #unlink(paste0(out_dir,"/*.csv"))
    #unlink(paste0(out_dir,"/*.asciipb"))
  }
  return(out)
}
