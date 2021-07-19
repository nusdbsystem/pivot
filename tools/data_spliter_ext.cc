#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;
#define MAX_CLIENT 10

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "Args: file_path" << endl;
    return 0;
  }

  auto file = argv[1];
  int num_client = atoi(argv[2]);
  ifstream ifs(file);
  string line;
  vector<vector<string>> table;
  while (getline(ifs, line)) {
    stringstream ss(line);
    string data;
    vector<string> sample;
    while (getline(ss, data, ',')) {
      // if (data == "0") data = "0.00001"; // temporarily fix zero value issue
      sample.push_back(data);
    }
    table.push_back(sample);
  }
  int n_samples = table.size();
  int n_attributes = table[0].size();
  cout << "number of samples: " << n_samples << endl
       << "number of attributes (including label): " << n_attributes << endl;

  int n_attr_per_client = n_attributes / num_client;

  std::vector<std::string> files;
  ofstream outs[MAX_CLIENT];
  for (int i = 0; i < num_client; i++) {
      std::string f = "client_" + to_string(i) + ".txt";
      files.push_back(f);
      outs[i].open(f);
      outs[i].precision(6);
  }

  int threshold_index[MAX_CLIENT];
  for (int i = num_client - 1; i >= 0; i--) {
      int threshold = n_attr_per_client * (num_client - i);
      threshold_index[i] = threshold;
  }
  threshold_index[num_client] = 0;
  threshold_index[0] = threshold_index[0] + 1; // include label

  for (auto entry : table) {
      for (int i = 0; i < n_attributes; ++i) {
          for (int client_id = num_client - 1; client_id >= 0; client_id --) {
              if (i < threshold_index[client_id] && i >= threshold_index[client_id + 1]) {
                  double d = std::stod(entry[i]);
                  outs[client_id] << fixed << d;
                  if (i == threshold_index[client_id] - 1) {
                      outs[client_id] << endl;
                  } else {
                      outs[client_id] << ",";
                  }
              }
          }
      }
  }

  /*
  string f1 = "client_0.txt", f2 = "client_1.txt", f3 = "client_2.txt";
  ofstream out1(f1), out2(f2), out3(f3);
  out1.precision(6);
  out2.precision(6);
  out3.precision(6);
  //out1 << std::fixed << std::setprecision(6);
  //out2 << std::fixed << std::setprecision(6);
  //out3 << std::fixed << std::setprecision(6);
  for (auto entry : table) {
    // out1 << "1.0,"; // x0 for regression
    for (int i = 0; i < n_attributes; ++i) {
      if (i < n_attr_per_client) {
        double d = std::stod(entry[i]);
        out3 << fixed << d;
        if (i == n_attr_per_client - 1) {
          out3 << endl;
        } else {
          out3 << ",";
        }
      } else if (i < n_attr_per_client * 2) {
        double d = std::stod(entry[i]);
        out2 << fixed << d;
        if (i == n_attr_per_client * 2 - 1) {
          out2 << endl;
        } else {
          out2 << ",";
        }
      } else {
        double d = std::stod(entry[i]);
	    out1 << fixed << d;
        if (i == n_attributes - 1) {
          out1 << endl;
        } else {
          out1 << ",";
        }
      }
    }
  }
  */
  return 0;
}
