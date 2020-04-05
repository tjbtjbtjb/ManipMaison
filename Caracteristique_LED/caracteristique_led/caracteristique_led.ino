
const int analogRef = A0;
const int analogMed = A2;
int ref,med;

const unsigned long delta_t=1000;

void setup() {
  Serial.begin(115200);
}

void loop() {
  
  while ( millis() < delta_t ) {
    ref = analogRead(analogRef);
    med = analogRead(analogMed);  
    Serial.print(ref); Serial.print(" "); Serial.println(med);
    delay(5);
  }
  
  Serial.println("Fini.");
    
  while(1) {
    /* c'est fini */
    delay(1000);
  }

}
