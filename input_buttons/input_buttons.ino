/*
The circuit for each button:
 * pushbutton attached to corresopnding pin from +5V
 * 10K resistor attached to corresponding pin from ground
*/

const int buttonPin0 = 4;
const int buttonPin1 = 5;
const int buttonPin2 = 6;
const int buttonPin3 = 7;
const int buttonPin4 = 8;
const int buttonPin5 = 9;
const int buttonPin6 = 10;

int buttonState0 = 0;
int buttonState1 = 0;
int buttonState2 = 0;
int buttonState3 = 0;
int buttonState4 = 0;
int buttonState5 = 0;
int buttonState6 = 0;

byte out = 0b00000000;

void setup() {
  pinMode(buttonPin0, INPUT);
  pinMode(buttonPin1, INPUT);
  pinMode(buttonPin2, INPUT);
  pinMode(buttonPin3, INPUT);
  pinMode(buttonPin4, INPUT);
  pinMode(buttonPin5, INPUT);
  pinMode(buttonPin6, INPUT);

  
}

void loop() {
  //button 0
  if (buttonState0 == HIGH) {
    
  }
  else {
    
  }
  //button 1
  if (buttonState1 == HIGH) {
    
  }
  else {
    
  }
//button 2
  if (buttonState2 == HIGH) {
    
  }
  else {
    
  }
//button 3
  if (buttonState3 == HIGH) {
    
  }
  else {
    
  }
//button 4
  if (buttonState4 == HIGH) {
    
  }
  else {
    
  }
//button 5
  if (buttonState5 == HIGH) {
    
  }
  else {
    
  }
//button 6
  if (buttonState6 == HIGH) {
    
  }
  else {
    
  }
}
