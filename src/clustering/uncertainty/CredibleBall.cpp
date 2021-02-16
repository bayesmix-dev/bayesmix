#include "CredibleBall.hpp"

using namespace std;

void transform_row2vec(Eigen::MatrixXi matrix, int row, Eigen::VectorXi vec) {
  // we save the specified row of matrix in vec
  if (row > matrix.rows()) {
    cout << "Error, row out of bounds!" << endl;
  }
  for (int j = 0; j < matrix.cols(); j++) {
    vec(j) = matrix(row, j);
  }

  return;
}

void add_row2matrix(Eigen::MatrixXi trg, int row, Eigen::MatrixXi src,
                    int index) {
  // we save in trg(row) the row src(index)
  int col = src.cols();

  for (int i = 0; i < col; i++) {
    trg(row, i) = src(index, i);
  }

  return;
}

CredibleBall::CredibleBall(LOSS_FUNCTION loss_type,
                           Eigen::MatrixXi& mcmc_sample_, double alpha_,
                           Eigen::VectorXi& point_estimate_)
    : loss_function(0) {
  cout << "[CONSTRUCTORS]" << endl;
  cout << "CredibleBall Constructor" << endl;
  switch (loss_type) {
    case BINDER_LOSS:
      loss_function = new BinderLoss(1, 1);
      break;
    case VARIATION_INFORMATION: {
      loss_function = new VariationInformation(false);
      break;
    }
    case VARIATION_INFORMATION_NORMALIZED: {
      loss_function = new VariationInformation(true);
      break;
    }
    default:
      throw std::domain_error("Loss function not recognized");
  }
  mcmc_sample = mcmc_sample_;
  T = mcmc_sample.rows();
  N = mcmc_sample.cols();
  alpha = alpha_;
  point_estimate = point_estimate_;
}

CredibleBall::~CredibleBall() {
  cout << "[Destructor]" << endl;
  cout << "CredibleBall Destructor" << endl;
  delete loss_function;
}

double CredibleBall::calculateRegion(double rate) {
  double epsilon = 1.0;
  double probability = 0.0;
  double steps = 1;
  loss_function->SetFirstCluster(point_estimate);
  cout << "Assessing the value of the radius..."
       << "\n";

  while (1) {
    epsilon = epsilon + rate * steps;
    cout << "Epsilon: " << epsilon << endl;

    for (int i = 0; i < T; i++) {
      Eigen::VectorXi vec;
      vec = mcmc_sample.row(i);
      loss_function->SetSecondCluster(vec);
      double loss = loss_function->Loss();
      if (loss <= epsilon) {
        probability += 1;
      }
    }

    probability /= T;

    if (probability >= 1 - alpha) {
      cout << "Probability: " << probability << endl;
      cout << "Radius estimated: " << epsilon << endl;
      radius = epsilon;
      populateCredibleSet();
      break;
    }

    steps++;
    probability = 0;
  }

  return epsilon;
}

void CredibleBall::populateCredibleSet() {
  loss_function->SetFirstCluster(point_estimate);
  // int count = 0;

  for (int i = 0; i < T; i++) {
    Eigen::VectorXi vec;
    vec = mcmc_sample.row(i);
    //  transform_row2vec(mcmc_sample.cast<int>(), i, vec);
    loss_function->SetSecondCluster(vec);

    if (loss_function->Loss() <= radius) {
      // cout << "populateCredibleSet1" << endl;
      // save the index of cluster belonging to the credible-ball
      credibleBall.insert(i);
    }
  }

  cout << "Intern data populated" << endl;
  cout << "Size of credible ball: (" << credibleBall.size() << ")" << endl;

  return;
}

int CredibleBall::count_cluster_row(int row) {
  // Returns the number of partitions in a specified row of the mcmc_sample
  if (row > T) {
    cout << "Row out of bounds!" << endl;
    return -1;
  }

  set<int, greater<int>> s;

  for (int i = 0; i < N; i++) {
    int tmp = mcmc_sample(row, i);
    s.insert(tmp);
  }

  return s.size();
}

//* clusters with min cardinality that are as distant as possible from the
//* center
Eigen::VectorXi CredibleBall::VerticalUpperBound() {
  set<int, greater<int>> vec1;  // vec1 is for saving the indexes of clusters
                                // that have min cardinality
  Eigen::VectorXi vec2;         // vec2 is an auxiliary vector
  set<int, greater<int>> vub;
  int tmp1;
  int min = INT_MAX;

  // find the minimal cardinality among the clusters
  for (auto i : credibleBall) {
    tmp1 = count_cluster_row(i);
    if (tmp1 <= min) {
      min = tmp1;
    }
  }

  // save the indexes of the clusters with min cardinality
  for (auto i : credibleBall) {
    if (count_cluster_row(i) == min) {
      vec1.insert(i);
    }
  }

  // among the clusters with min cardinality find the max distance
  // from the point_estimate
  double max_distance = -1.0;
  for (auto i : vec1) {
    // we save the row "vec(i)" of the credibleBall in vec2
    // compute the distance, ie the loss for the corresponding row
    loss_function->SetFirstCluster(point_estimate);
    vec2 = mcmc_sample.row(i);
    loss_function->SetSecondCluster(vec2);
    double loss = loss_function->Loss();

    if (loss > max_distance) {
      max_distance = loss;
    }
  }

  // save the clusters that are "max_distance" far away from the point estimate
  for (auto i : vec1) {
    loss_function->SetFirstCluster(point_estimate);
    vec2 = mcmc_sample.row(i);
    loss_function->SetSecondCluster(vec2);
    double loss = loss_function->Loss();

    if (loss == max_distance) {
      vub.insert(i);
    }
  }

  Eigen::VectorXi VUB(vub.size());  // the output
  int aux = 0;
  for (auto i : vub) {
    VUB(aux) = i;
    aux++;
  }

  return VUB;
}

//* clusters with max cardinality that are as distant as possible from the
//* center
Eigen::VectorXi CredibleBall::VerticalLowerBound() {
  set<int, greater<int>> vec1;
  Eigen::VectorXi vec2;
  // vec1 has the indexes of the clusters with maximum cardinality in the ball
  // vec2 is an auxiliary vector
  set<int, greater<int>> vlb;

  // int rows = credibleBall.size();
  int tmp1;
  int max = -1;

  // find the maximal cardinality among the clusters
  for (auto i : credibleBall) {
    tmp1 = count_cluster_row(i);
    if (tmp1 >= max) {
      max = tmp1;
    }
  }

  // save the indexes of the clusters with max cardinality
  for (auto i : credibleBall) {
    if (count_cluster_row(i) == max) {
      vec1.insert(i);
    }
  }

  // among the clusters with max cardinality find the max distance
  // from the point_estimate
  double max_distance = -1.0;
  for (auto i : vec1) {
    loss_function->SetFirstCluster(point_estimate);
    vec2 = mcmc_sample.row(i);
    loss_function->SetSecondCluster(vec2);
    double loss = loss_function->Loss();

    if (loss > max_distance) {
      max_distance = loss;
    }
  }

  // save the clusters that are "max_distance" far away from the point estimate
  for (auto i : vec1) {
    loss_function->SetFirstCluster(point_estimate);
    vec2 = mcmc_sample.row(i);
    loss_function->SetSecondCluster(vec2);
    double loss = loss_function->Loss();

    if (loss == max_distance) {
      vlb.insert(i);
    }
  }

  Eigen::VectorXi VLB(vlb.size());
  int aux = 0;
  for (auto i : vlb) {
    VLB(aux) = i;
    aux++;
  }
  return VLB;
}

//* clusters in the credible ball that are more distant from the center
Eigen::VectorXi CredibleBall::HorizontalBound() {
  set<int, greater<int>> hb;
  Eigen::VectorXi vec;
  double max_distance = -1.0;

  // find the max distance among all the clusters in the credible ball
  for (auto i : credibleBall) {
    loss_function->SetFirstCluster(point_estimate);
    vec = mcmc_sample.row(i);
    loss_function->SetSecondCluster(vec);
    double loss = loss_function->Loss();

    if (loss >= max_distance) {
      max_distance = loss;
    }
  }

  // select the clusters with that distance
  for (auto i : credibleBall) {
    loss_function->SetFirstCluster(point_estimate);
    vec = mcmc_sample.row(i);
    // transform_row2vec(credibleBall, i, vec);
    loss_function->SetSecondCluster(vec);
    double loss = loss_function->Loss();

    if (loss == max_distance) {
      hb.insert(i);
    }
  }

  Eigen::VectorXi HB(hb.size());
  int aux = 0;
  for (auto i : hb) {
    HB(aux) = i;
    aux++;
  }
  return HB;
}
