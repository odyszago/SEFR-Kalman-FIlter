
const byte FEATURES   = 4;   // number of features
const byte LABELS     = 2;   // number of labels
const byte DATAFACTOR = 10;  // scale factor of data

const unsigned int DATASET_MAXSIZE = 50; //  dataset size 
unsigned int       dataset_size    = 50; // current dataset size


float DATASET[DATASET_MAXSIZE][FEATURES] = {
 
{18.45, 49.75,  31.98,  1019.98},
{20.81, 54.28,  29.22,  1017.84},
{19.02, 48.61,  30.83,  1020.33},
{20.81, 51.08,  30.01,  1014.55},
{18.17, 57.83,  29.87,  1009.24},
{19.51, 46.26,  30.18,  1017.79},
{18.08, 52.32,  29.36,  1014.99},
{21.01, 52.75,  29.4  ,1014.84},
{21.15, 48.44,  32.94,  1014.19},
{21.41, 52.83,  32.21,  1020.1},
{19.4 ,59.49  ,30.98  ,1011.7},
{18.6 ,48.07  ,32.58  ,1015.48},
{22.81, 48.35,  31.52,  1020.93},
{19.79, 52.64,  29.38,  1018.35},
{19.85, 47.29,  30.12,  1009.63},
{20.69, 59.43,  31.86,  1014.35},
{18.86, 49.58,  32.21,  1017.8},
{20.98, 45.37,  32.11,  1013.62},
{21.19, 55.52,  30.56,  1020.53},
{18.4 ,57.12  ,30.96  ,1015.96},
{18.5 ,55.55  ,31.63  ,1010.52},
{20.09, 48.83,  29.41,  1019.86},
{22.57, 59.14,  31.76,  1021.49},
{18.86, 47.1  ,32.58  ,1009.67},
{21.96, 46.85 ,31.09, 1013.85},
{20.97, 54.17,  31.0  ,1009.61},
{18.58, 59.23,  32.11,  1018.05 },
{20.66, 47.2  ,29.84  ,1015.07 },
{22.18, 57.28,  32.86,  1019.81 },
{22.21, 48.33,  32.94,  1018.47 },
{22.62, 58.28,  32.74,  1011.64 },
{19.01, 59.12,  30.57,  1011.4 },
{19.1 ,51.39  ,29.31  ,1015.07 },
{22.62, 52.93,  30.17,  1012.53 },
{18.01, 48.41,  30.15,  1012.54 },
{20.25, 48.04,  31.98,  1019.63 },
{19.08, 51.04,  30.31,  1016.95 },
{20.84, 55.29,  31.54,  1012.84 },
{18.92, 55.31,  30.33,  1014.85 },
{22.58, 47.03,  30.38,  1018.68 },
{21.56, 59.1  ,29.69  ,1021.0 },
{22.49, 49.61,  29.42,  1014.06 },
{22.31, 54.01,  30.93,  1012.25 },
{22.93, 49.83,  29.91,  1012.28 },
{20.69, 47.19,  31.12,  1011.93 },
{21.55, 48.71,  30.77,  1014.65 },
{20.31, 54.2  ,32.59  ,1016.9},
{ 22.57,  57.23,  29.25,  1017.93 },
{21.7 ,50.12  ,32.15  ,1014.79},
{19.63, 45.47,  29.98,  1017.17 }

};

// labels of the  dataset
byte TARGET[DATASET_MAXSIZE] = {
1,
0,
1,
1,
0,
1,
1,
0,
1,
0,
0,
1,
1,
0,
1,
0,
1,
1,
0,
0,
0,
1,
0,
1,
1,
0,
0,
1,
0,
1,
0,
0,
1,
0,
1,
1,
1,
0,
0,
1,
0,
1,
0,
1,
1,
1,
0,
0,
1,
1




};

float weights[LABELS][FEATURES]; // model weights on each labels
float bias[LABELS];              // model bias on each labels


// ==================================================

// train SEFR model with DATASET and TARGET
void fit() {
  unsigned int training_time;      // mode training time
  unsigned int start_time = millis();

  // iterate all labels
  for (byte l = 0; l < LABELS; l++) {

    unsigned int count_pos = 0, count_neg = 0;

    // iterate all features
    for (byte f = 0; f < FEATURES; f++) {

      float avg_pos = 0.0, avg_neg = 0.0;
      count_pos = 0;
      count_neg = 0;
      for (unsigned int s = 0; s < dataset_size; s++) {
        if (TARGET[s] != l) { // use "not the label" as positive class
          avg_pos += float(DATASET[s][f]);
          count_pos++;
        } else { // use the label as negative class
          avg_neg += float(DATASET[s][f]);
          count_neg++;
        }
      }
      avg_pos /= (float(count_pos) * float(DATAFACTOR));
      avg_neg /= (float(count_neg) * float(DATAFACTOR));

      // calculate weight of this label
      weights[l][f] = (avg_pos - avg_neg) / (avg_pos + avg_neg);
    }

    // calculate average weighted score for positive/negative data
    float avg_pos_w = 0.0, avg_neg_w = 0.0;
    for (unsigned int s = 0; s < dataset_size; s++) {
      float weighted_score = 0.0;
      for (byte f = 0; f < FEATURES; f++) {
        weighted_score += (float(DATASET[s][f]) * weights[l][f]);
      }
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
  training_time = millis() - start_time;
  Serial.println("training_time : ");
  Serial.println(training_time);

}

// predict label from a single new data instance
byte predict(float new_data[FEATURES]) {

  float score[LABELS];
  for (byte l = 0; l < LABELS; l++) {
    score[l] = 0.0;
    for (byte f = 0; f < FEATURES; f++) {
      // calculate weight of each labels
      score[l] += (float(new_data[f]) / float(DATAFACTOR) * weights[l][f]);
    }
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
