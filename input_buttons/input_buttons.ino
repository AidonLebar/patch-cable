/*
The circuit for each button:
 * pushbutton attached to corresopnding pin from +5V
 * 10K resistor attached to corresponding pin from ground
 * potentiometer to analog pin 3
*/

const int buttonPin0 = 4;
const int buttonPin1 = 5;
const int buttonPin2 = 6;
const int buttonPin3 = 7;
const int buttonPin4 = 8;
const int buttonPin5 = 9;
const int buttonPin6 = 10;
const int analogPin = 3;

int buttonState0 = 0;
int buttonState1 = 0;
int buttonState2 = 0;
int buttonState3 = 0;
int buttonState4 = 0;
int buttonState5 = 0;
int buttonState6 = 0;
int potVal = 0;

byte out = 0b00000000;

void setup() {
  pinMode(buttonPin0, INPUT);
  pinMode(buttonPin1, INPUT);
  pinMode(buttonPin2, INPUT);
  pinMode(buttonPin3, INPUT);
  pinMode(buttonPin4, INPUT);
  pinMode(buttonPin5, INPUT);
  pinMode(buttonPin6, INPUT);

  Serial.begin(9600);
}

void loop() { 
  // evil bit level hacking
  // what the fuck?
  
  //button 0
  buttonState0 = digitalRead(buttonPin0);
  if (buttonState0 == HIGH) {
    out |= 0b01000000;
  }
  else {
    out &= 0b10111111;
  }
  //button 1
  buttonState1 = digitalRead(buttonPin1);
  if (buttonState1 == HIGH) {
    out |= 0b00100000;
  }
  else {
    out &= 0b11011111;
  }
  //button 2
  buttonState2 = digitalRead(buttonPin2);
  if (buttonState2 == HIGH) {
    out |= 0b00010000;
  }
  else {
    out &= 0b11101111;
  }
  //button 3
  buttonState3 = digitalRead(buttonPin3);
  if (buttonState3 == HIGH) {
    out |= 0b00001000;
  }
  else {
    out &= 0b11110111;
  }
  //button 4
  buttonState4 = digitalRead(buttonPin4);
  if (buttonState4 == HIGH) {
    out |= 0b00000100;
  }
  else {
    out &= 0b11111011;
  }
  //button 5
  buttonState5 = digitalRead(buttonPin5);
  if (buttonState5 == HIGH) {
    out |= 0b00000010;
  }
  else {
    out &= 0b11111101;
  }
  //button 6
  buttonState6 = digitalRead(buttonPin6);
  if (buttonState6 == HIGH) {
    out |= 0b00000001;
  }
  else {
    out &= 0b11111110;
  }

  int potVal = analogRead(analogPin);

  delay(100);
  Serial.write(out); //send input back to computer
  Serial.println(potVal);
}
