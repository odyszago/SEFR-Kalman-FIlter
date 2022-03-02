const byte LABELS     = 2;   // number of labels
const byte DATAFACTOR = 10;  // scale factor of data

const unsigned int DATASET_MAXSIZE = 50; // dataset size 
unsigned int       dataset_size    = 50; // current dataset size

// the fused dataset (times DATAFACTOR so it can be stored as integer and save space/memory)
float DATASET[DATASET_MAXSIZE] = {
 254.39,
245.81,
246.18,
243.87,
242.85,
250.14,
248.99,
247.8,
247.75,
246.43,
244.61,
250.98,
249.32,
246.99,
244.67,
243.96,
249.62,
248.06,
244.17,
243.38,
242.88,
246.29,
246.65,
251.04,
250.82,
245.79,
247.6,
250.52,
251.52,
254.46,
251.41,
250.79,
250.31,
246.54,
247.8,
249.69,
248.65,
245.29,
246.01,
250.52,
250.53,
252.29,
251.07,
251.64,
251.83,
250.65,
251.33,
248.28,
247.95,
249.76

};


// labels of the fused dataset
byte TARGET[DATASET_MAXSIZE] = {
 1,
0,
0,
0,
0,
1,
1,
0,
0,
0,
0,
1,
1,
0,
0,
0,
1,
1,
0,
0,
0,
0,
0,
1,
1,
0,
0,
1,
1,
1,
1,
1,
1,
0,
0,
1,
1,
0,
0,
1,
1,
1,
1,
1,
1,
1,
1,
1,
0,
1

};


float weights[LABELS]; // model weights on each labels
float bias[LABELS];   // model bias on each labels

  

// ==================================================

// train SEFR model with DATASET and TARGET
void fit() {

  Serial.println("in fit before");
  unsigned long start_time = millis();
  Serial.println("in fit after");
  // iterate all labels
  for (byte l = 0; l < LABELS; l++) {

    unsigned int count_pos = 0, count_neg = 0;
      float avg_pos = 0.0, avg_neg = 0.0;
      count_pos = 0;
      count_neg = 0;
      for (unsigned int s = 0; s < dataset_size; s++) {
        if (TARGET[s] != l) { // use "not the label" as positive class
          avg_pos += float(DATASET[s]);
          count_pos++;
        } else { // use the label as negative class
          avg_neg += float(DATASET[s]);
          count_neg++;
        }
      }
      avg_pos /= (float(count_pos) * float(DATAFACTOR));
      avg_neg /= (float(count_neg) * float(DATAFACTOR));

      
      weights[l] = (avg_pos - avg_neg) / (avg_pos + avg_neg);


    // calculate average weighted score for positive/negative data
    float avg_pos_w = 0.0, avg_neg_w = 0.0;
    for (unsigned int s = 0; s < dataset_size; s++) {
      float weighted_score = 0.0;
  
        weighted_score += (float(DATASET[s]) * weights[l]);
      
      if (TARGET[s] != l) {
        avg_pos_w += weighted_score;
      } else {
        avg_neg_w += weighted_score;
      }
    }
    avg_pos_w /= (float(count_pos) * float(DATAFACTOR));
    avg_neg_w /= (float(count_neg) * float(DATAFACTOR));

    // calculate bias of this label
    bias[l] = -1 * (float(count_neg) * avg_pos_w + float(count_pos) * avg_neg_w) / float(count_pos + count_neg);
  }

  // calculate training time
 unsigned long training_time = millis() - start_time;
  Serial.println("Trained:");
  Serial.print(training_time);
   Serial.println(" ms");
  delay(2000);

}

// predict label from a single new data instance
byte predict(float new_data) {

  float score[LABELS];
  for (byte l = 0; l < LABELS; l++) {
    score[l] = 0.0;

      score[l] += (float(new_data) / float(DATAFACTOR) * weights[l]);
  
    score[l] += bias[l]; // add bias of each labels
  }

  // find the min score (least possible label of "not the label")
  float min_score = score[0];
  byte min_label = 0;
  for (byte l = 1; l < LABELS; l++) {
    if (score[l] < min_score) {
      min_score = score[l];
      min_label = l;
    }
  }

  return min_label; // return prediction

}
