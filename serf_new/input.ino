#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#define LED 4
#define SEALEVELPRESSURE_HPA (1013.25)
Adafruit_BME280 bme; // I2C
void setup() {
  digitalWrite(LED,LOW);
  Serial.begin(9600);
  unsigned status = bme.begin();  
    fit();
}

void loop() { 

     Serial.println("New input");
     //Temperature *C, Pressure hPa ,Approx. Altitude m  ,Humidity %
    float temp = bme.readTemperature();
    Serial.print(temp);
    Serial.print(",");
    float hum = bme.readHumidity();
    Serial.print(hum);
    Serial.print(",");
    float alt = bme.readAltitude(SEALEVELPRESSURE_HPA);
    Serial.print(alt);
    Serial.print(",");
    float pressure = bme.readPressure() / 100.0F;
    Serial.print(pressure);
    Serial.println();
    delay(1000);

    float data[4] = {temp,hum,alt,pressure};
       
    byte pred = predict(data);
    Serial.println("predict");
    Serial.println(pred);

    if ( pred == 0 )
      digitalWrite(LED,LOW);
    else
      digitalWrite(LED,HIGH);
    delay(5000);
    digitalWrite(LED,LOW);
      
}
