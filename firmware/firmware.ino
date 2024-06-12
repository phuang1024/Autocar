#include <IBusBM.h>


class Motors {
public:
    Motors() {
        for (int i = 4; i < 8; i++) {
            pinMode(i, OUTPUT);
            digitalWrite(i, LOW);
        }
        for (int i = 22; i < 30; i++) {
            pinMode(i, OUTPUT);
            digitalWrite(i, LOW);
        }
    }

    void write_dir(int i, int dir) {
        int pin = 22 + 2 * i;
        if (i >= 2) {
            dir = !dir;
        }
        digitalWrite(pin, dir);
        digitalWrite(pin + 1, !dir);
    }

    void write_speed(int i, int speed) {
        int pin = 4 + i;
        analogWrite(pin, speed);
    }
};


Motors motors;
IBusBM ibus;


void setup() {
    delay(100);

    Serial.begin(9600);
    ibus.begin(Serial1, 2);

    for (int i = 0; i < 4; i++) {
        motors.write_dir(i, true);
    }
    while (true) {
        // read RC
        uint16_t rx_steering = ibus.readChannel(0);
        uint16_t rx_throttle = ibus.readChannel(2);
        uint16_t rx_enable = ibus.readChannel(4);

        int steer = map(rx_steering, 1000, 2000, -50, 50);
        int speed = map(rx_throttle, 1000, 2000, 0, 255);
        int speed1 = speed + steer;
        int speed2 = speed - steer;
        Serial.println(rx_steering);
        Serial.println(steer);
        speed1 = constrain(speed1, 0, 255);
        speed2 = constrain(speed2, 0, 255);

        if (rx_enable > 1500) {
            motors.write_speed(0, speed1);
            motors.write_speed(1, speed1);
            motors.write_speed(2, speed2);
            motors.write_speed(3, speed2);
        } else {
            for (int i = 0; i < 4; i++) {
                motors.write_speed(i, 0);
            }
        }

        delay(50);
    };
}


void loop() {
}
