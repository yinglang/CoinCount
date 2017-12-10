template <typename Dtype>
void SSDTestDetection(const Dtype*& result_vec, int size, int num_det) {
  map<int, map<int, vector<pair<float, int> > > > all_true_pos;
  map<int, map<int, vector<pair<float, int> > > > all_false_pos;
  map<int, map<int, int> > all_num_pos;
  Dtype loss = 0;
    for (int j = 0; j < result.size(); ++j) {
      CHECK_EQ(result[j]->width(), 5);
      const Dtype* result_vec = result[j]->cpu_data();
      int num_det = result[j]->height();
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);
        int label = static_cast<int>(result_vec[k * 5 + 1]);
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];
          int tp = static_cast<int>(result_vec[k * 5 + 3]);
          int fp = static_cast<int>(result_vec[k * 5 + 4]);
          if (tp == 0 && fp == 0) {
            // Ignore such case. It happens when a detection bbox is matched to
            // a difficult gt bbox and we don't evaluate on difficult gt bbox.
            continue;
          }
          all_true_pos[j][label].push_back(std::make_pair(score, tp));
          all_false_pos[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
}
  
  for (int i = 0; i < all_true_pos.size(); ++i) {
    if (all_true_pos.find(i) == all_true_pos.end()) {
      LOG(FATAL) << "Missing output_blob true_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& true_pos =
        all_true_pos.find(i)->second;
    if (all_false_pos.find(i) == all_false_pos.end()) {
      LOG(FATAL) << "Missing output_blob false_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& false_pos =
        all_false_pos.find(i)->second;
    if (all_num_pos.find(i) == all_num_pos.end()) {
      LOG(FATAL) << "Missing output_blob num_pos: " << i;
    }
    const map<int, int>& num_pos = all_num_pos.find(i)->second;
    map<int, float> APs;
    float mAP = 0.;
    // Sort true_pos and false_pos with descend scores.
    for (map<int, int>::const_iterator it = num_pos.begin();
         it != num_pos.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (true_pos.find(label) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_true_pos =
          true_pos.find(label)->second;
      if (false_pos.find(label) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_false_pos =
          false_pos.find(label)->second;
      vector<float> prec, rec;
      ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                param_.ap_version(), &prec, &rec, &(APs[label]));

      std::string filename = "/root/Workspace/caffe-ssd/AP/ap.txt";
      PrintToFile(filename, prec, rec, label_true_pos);

      mAP += APs[label];
      if (param_.show_per_class_result()) {
        LOG(INFO) << "class" << label << ": " << APs[label];
      }
    }
    mAP /= num_pos.size();
    const int output_blob_index = test_net->output_blob_indices()[i];
    const string& output_name = test_net->blob_names()[output_blob_index];
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mAP;
  
}
