#include "plot_utils.h"

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
to_mesh(const Eigen::MatrixXd &grid, const Eigen::VectorXd &vals) {
  // infer the number of points in the ygrid
  int ny = 0;
  double first_x = grid(0, 0);
  while (ny < grid.rows() && grid(ny + 1, 0) == first_x) {
    ny += 1;
  }
  ny += 1;

  int nx;
  if (grid.rows() % ny == 0) {
    nx = grid.rows() / ny;
  } else {
    throw std::invalid_argument("grid is not regular");
  }

  std::vector<std::vector<double>> xgrid(ny, std::vector<double>(nx, 0));
  std::vector<std::vector<double>> ygrid(nx, std::vector<double>(ny, 0));
  std::vector<std::vector<double>> zgrid(ny, std::vector<double>(nx, 0));

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      xgrid[i][j] = grid(i * ny + j, 0);
      ygrid[i][j] = grid(i * ny + j, 1);
      zgrid[i][j] = vals(i * ny + j);
    }
  }
  return std::make_tuple(xgrid, ygrid, zgrid);
}

void density_plot_1d(const Eigen::MatrixXd &grid, const Eigen::VectorXd &dens,
                     const std::string &outfile) {
  int n_points = grid.size();
  std::vector<double> grid_vec(grid.data(), grid.data() + n_points);
  std::vector<double> mean_dens_vec(dens.data(), dens.data() + n_points);
  matplot::plot(grid_vec, mean_dens_vec);
  matplot::title("Density estimate");
  matplot::xlabel("Grid");
  matplot::ylabel("Density");
  matplot::save(outfile);
  std::cout << "Saved density plot to " << outfile << std::endl;
}

void density_plot_2d(const Eigen::MatrixXd &grid, const Eigen::VectorXd &dens_,
                     const std::string &outfile, bool log_scale /*= true*/) {
  int n_points = grid.size();

  Eigen::VectorXd dens = dens_;
  if (log_scale) {
    dens = dens_.array().log();
  } else {
    dens = dens_ * 10;
  }
  auto [X, Y, Z] = to_mesh(grid, dens);

  matplot::contour(X, Y, Z)->line_width(2);
  matplot::hold(false);

  matplot::title("Density estimate");
  matplot::xlabel("X");
  matplot::ylabel("Y");
  matplot::save(outfile);
  std::cout << "Saved density plot to " << outfile << std::endl;
}

void num_clus_trace(const Eigen::MatrixXd &num_clus_chain,
                    const std::string &outfile) {
  int n_iters = num_clus_chain.size();
  std::vector<double> num_clus_vec(num_clus_chain.data(),
                                   num_clus_chain.data() + n_iters);
  std::vector<double> iters_vec(n_iters);
  std::iota(iters_vec.begin(), iters_vec.end(), 1);

  matplot::plot(iters_vec, num_clus_vec);
  matplot::hold(true);
  matplot::scatter(iters_vec, num_clus_vec);
  matplot::hold(false);
  matplot::title("Traceplot of number of clusters from the MCMC");
  matplot::xlabel("MCMC iterations");
  matplot::ylabel("Number of clusters");
  matplot::save(outfile);
  std::cout << "Saved traceplot to " << outfile << std::endl;
}

void num_clus_bar(const Eigen::MatrixXd &num_clus_chain_,
                  const std::string &outfile) {
  int n_iters = num_clus_chain_.size();
  const Eigen::VectorXi &num_clus_chain = num_clus_chain_.col(0).cast<int>();
  int xmin = num_clus_chain.minCoeff();
  int xmax = num_clus_chain.maxCoeff();
  std::vector<int> xticks(xmax - xmin + 1);
  std::iota(xticks.begin(), xticks.end(), xmin);

  Eigen::VectorXd bar_heights_ = Eigen::VectorXd::Zero(xticks.size());
  for (int i = 0; i < n_iters; i++) {
    bar_heights_[num_clus_chain[i] - xmin] += 1;
  }
  bar_heights_ = bar_heights_.array() / n_iters;

  std::vector<double> bar_heights(bar_heights_.data(),
                                  bar_heights_.data() + bar_heights_.size());

  matplot::bar(xticks, bar_heights);
  matplot::title("Posterior number of clusters");
  matplot::xlabel("Number of clusters");
  matplot::ylabel("Probability");
  matplot::save(outfile);
  std::cout << "Saved barplot to " << outfile << std::endl;
}
