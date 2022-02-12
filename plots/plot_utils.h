#ifndef BAYESMIX_PLOTS_PLOT_UTILS_H_
#define BAYESMIX_PLOTS_PLOT_UTILS_H_

#include <matplot/matplot.h>

#include <Eigen/Dense>
#include <numeric>

/*
 * Converts the support points of a 2d function and associated values
 * from the format {(x_i, y_i), z_i} stored in grid and vals respectively,
 * to grids over the 2d domain. Used in density_plot_2d.
 */
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
to_mesh(const Eigen::MatrixXd &grid, const Eigen::VectorXd &vals);

void density_plot_1d(const Eigen::MatrixXd &grid, const Eigen::VectorXd &dens,
                     const std::string &outfile);

void density_plot_2d(const Eigen::MatrixXd &grid, const Eigen::VectorXd &dens_,
                     const std::string &outfile, bool log_scale = true);

void num_clus_trace(const Eigen::MatrixXd &num_clus_chain,
                    const std::string &outfile);

void num_clus_bar(const Eigen::MatrixXd &num_clus_chain_,
                  const std::string &outfile);

#endif  // BAYESMIX_PLOTS_PLOT_UTILS_H_
