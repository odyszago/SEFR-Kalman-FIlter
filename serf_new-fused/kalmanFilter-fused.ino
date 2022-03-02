#include <BasicLinearAlgebra.h>

// All the functions in BasicLinearAlgebra are wrapped up inside the namespace BLA, so specify that we're using it like
// so:
using namespace BLA;

 BLA::Matrix<4, 4> A = {1.0, 0, 1.0, 0,0, 1.0, 0, 1.0,0, 0, 1.0, 0,0, 0, 0, 1.0};
 BLA::Matrix<4,1> x =  {0.312242,0.5803398,0,0};
 BLA::Matrix<4,1> ground_truth = {0,0,0,0};
 BLA::Matrix<4,4> P = {1, 0, 0, 0,0, 1, 0, 0,0, 0, 1000, 0,0, 0, 0, 1000};      
 BLA::Matrix<4,4> Q = {0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0};
 BLA::Matrix<2,1> z_lidar = {0,0};
     
void predict_kalman_filter() {
    // Predict Step
    x = A * x;
    BLA::Matrix<4, 4> At = ~A;
    At = P * At;
    At = A * At;
    P = At + Q;
  
    
}
void update_kalman_filter(){
  
  BLA::Matrix<2,4> H = {1.0, 0, 0, 0,0, 1.0, 0, 0};
  BLA::Matrix<4,4> I = {1.0, 0, 0.0, 0,0, 1.0, 0, 0.0,0, 0, 1.0, 0,0, 0, 0, 1.0};   
  BLA::Matrix<2,2> R = {0.0225, 0,0, 0.0225} ; 
  //# Measurement update step
 
  BLA::Matrix<2,1> Y = H * x;
  
  Y = z_lidar - Y;

  BLA::Matrix<4, 2> Ht = ~H;
  BLA::Matrix<4,2> PHt = P * Ht;
  BLA::Matrix<2,2> HPHt = H * PHt; 
 
  BLA::Matrix<2,2> S = HPHt + R;
 
  BLA::Matrix<4,2> K = P * Ht;

  float detA = S(0,0)*S(1,1) - S(0,1)*S(1,0);
  BLA::Matrix<2,2> adjS;
  adjS.Fill(0); 
  adjS(0,0) = S(1,1);
  adjS(0,1) = -S(0,1);
  adjS(1,0) = -S(1,0);
  adjS(1,1) = S(0,0);
  BLA::Matrix<2,2> Si;
  Si.Fill(0); 
  Si(0,0) = adjS(0,0)/detA;
  Si(0,1) = adjS(0,1)/detA;
  Si(1,0) = adjS(1,0)/detA;
  Si(1,1) = adjS(1,1)/detA;
  K = K * Si;
  // New state

  BLA::Matrix<4,1> KY = K * Y;

  x = x + KY;
 


  BLA::Matrix<4,4> KH = K * H;

  BLA::Matrix<4,4> IKH = I - KH;
 
  P = IKH * P;

}

float CalculateRMSE() {
 
  BLA::Matrix <4> rmse;
  rmse.Fill(0);
  for (int i = 0; i < 4; i++){
    if (x(i,0) - ground_truth(i,0) >= 0)
        rmse(i) = x(i,0) - ground_truth(i,0);
     else 
        rmse(i) = -1.0 * (x(i,0) - ground_truth(i,0));
  }
  
  float sum = rmse(0) + rmse(1) + rmse(2) + rmse(3);
  return sum / 4.0;
  
}

double fused_value(float measurements[4]) {
    
    delay(500);
    Serial.println("Measurements");
    
    float dt = 2.0;
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;
    float noise_ax  =5;
    float noise_ay = 5;
    
    // Updating matrix A with dt value
    A(0,2) = dt;
    A(1,3) = dt;

    // Updating Q matrix
    Q(0,0) = dt_4 / 4 * noise_ax;
    Q(0,2) = dt_3 / 2 * noise_ax;
    Q(1,1) = dt_4 / 4 * noise_ay;
    Q(1,3) = dt_3 / 2 * noise_ay;
    Q(2,0) = dt_3 / 2 * noise_ax;
    Q(2,2) = dt_2 * noise_ax;
    Q(3,1) = dt_3 / 2 * noise_ay;
    Q(3,3) = dt_2 * noise_ay;

    // Updating sensor readings
    z_lidar(0,0) = (double) measurements[0];
    z_lidar(1,0) = (double) measurements[1];

    // Collecting ground truths
    ground_truth(0,0) = (double) measurements[2];
    ground_truth(1,0) = (double) measurements[3];
    
    //Call Kalman Filter Predict and Update functions.
    predict_kalman_filter();
  
    update_kalman_filter();
    
    float mean = CalculateRMSE();
    Serial.println("Result :");
    Serial.println(mean);
    delay(500);
    return mean;
}
