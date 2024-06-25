#include <IBusBM.h>

const float MIN_V = 3.1;
const float BATT_S = 3;


// pin order: ena, pul, dir
class Motors {
public:
    Motors() {
        for (int i = 2; i <= 7; i++) {
            pinMode(i, OUTPUT);
            digitalWrite(i, LOW);
        }
    }

    void write_ena(bool ena) {
        digitalWrite(2, ena);
        digitalWrite(5, ena);
    }

    void write_dir_left(bool dir) {
        digitalWrite(4, dir);
    }

    void write_dir_right(bool dir) {
        digitalWrite(7, dir);
    }

    void write_dir_both(bool dir) {
        write_dir_left(dir);
        write_dir_right(dir);
    }

    void write_pwm_left(int pwm) {
        analogWrite(3, 255 - pwm);
    }

    void write_pwm_right(int pwm) {
        analogWrite(6, 255 - pwm);
    }

    void write_pwm_both(int pwm) {
        write_pwm_left(pwm);
        write_pwm_right(pwm);
    }

    void write_vel_left(int vel) {
        write_dir_left(vel >= 0);
        write_pwm_left(abs(vel));
    }

    void write_vel_right(int vel) {
        write_dir_right(vel < 0);
        write_pwm_right(abs(vel));
    }

    void write_vel_both(int vel) {
        write_vel_left(vel);
        write_vel_right(vel);
    }
};


Motors motors;
IBusBM ibus;


float read_batt_v() {
    int raw = analogRead(A0);
    return (float)raw * 4 * 5 / 1023;
}


void setup() {
    delay(100);

    Serial.begin(9600);
    ibus.begin(Serial1, 1);

    unsigned long batt_low_time = 0;

    while (true) {
        // check voltage
        float batt_v = read_batt_v();
        bool batt_low = batt_v < MIN_V * BATT_S;
        if (batt_low) {
            batt_low_time = millis();
        }
        batt_low = batt_low || (millis() - batt_low_time) < 1000;

        // read RC
        uint16_t rx_steering = ibus.readChannel(0);
        uint16_t rx_throttle = ibus.readChannel(2);
        uint16_t rx_enable = ibus.readChannel(4);
        uint16_t rx_throttle2 = ibus.readChannel(1);

        int speed1 = map(rx_throttle, 1000, 2000, -255, 255);
        int speed2 = map(rx_throttle2, 1000, 2000, -255, 255);

        if (batt_low) {
            motors.write_ena(false);
        } else {
            motors.write_ena(rx_enable > 1500);
        }

        motors.write_vel_left(speed1);
        motors.write_vel_right(speed2);

        delay(10);
    }
}


void loop() {
}
