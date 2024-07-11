/*
Communication with serial: 115200 baud

Message:
ena v1 v2
v1, v2 are -255 to 255
Example:
1 0 255

Response:
batt rc1 rc2 rc3 rc4 rc5 rc6
batt is volts.
rc is 1000 to 2000.
Example:
10.4 1000 2000 1500 1234 1345 1234

*/


#include <IBusBM.h>


// pin order: ena, pul, dir
struct Motors {
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


struct BattV {
    const float MIN_V = 3.2;
    const float BATT_S = 3;
    const float LOW_THRES = MIN_V * BATT_S;
    // Voltage splitter factor.
    const float V_MULT = 4;
    // Maintain a "low" status for this long.
    const int LOW_COOLDOWN = 1000;

    unsigned long low_time;
    float voltage;

    unsigned long flash_time;
    bool flash_state;

    BattV() {
        pinMode(A0, INPUT);
        pinMode(13, OUTPUT);

        low_time = 0;
        voltage = 0;

        flash_time = 0;
        flash_state = true;
    }

    // Returns absolute voltage.
    float read() {
        int raw = analogRead(A0);
        return (float)raw * V_MULT * 5 / 1023;
    }

    // Returns is_low, taking into account the cooldown.
    bool update() {
        voltage = read();
        bool is_low = voltage < LOW_THRES;
        if (is_low) {
            // update flash
            if (millis() - flash_time > 1000) {
                flash_state = !flash_state;
                flash_time = millis();
                digitalWrite(13, flash_state);
            }

            low_time = millis();
            return true;
        } else {
            digitalWrite(13, HIGH);
        }
        return millis() - low_time < LOW_COOLDOWN;
    }
};


struct SerialQuery {
    bool active;
    bool ena;
    int v1, v2;

    // Read from Serial and initialize.
    void read() {
        if (Serial.available() > 0) {
            active = true;

            int ena_value = Serial.parseInt();
            ena = ena_value > 0;

            if (Serial.available() > 0) {
                v1 = Serial.parseInt();
                v2 = Serial.parseInt();
            }
        } else {
            active = false;
            ena = false;
        }
    }
};


Motors motors;
BattV batt;
IBusBM ibus;


void setup() {
    delay(100);

    Serial.begin(115200);
    ibus.begin(Serial1, 1);

    while (true) {
        // read serial messages
        SerialQuery query;
        query.read();

        // check voltage
        bool batt_low = batt.update();

        // read RC
        uint16_t rc_values[6];
        for (int i = 0; i < 6; i++) {
            rc_values[i] = ibus.readChannel(i);
        }

        // send response
        Serial.print(batt.voltage);
        Serial.print(" ");
        for (int i = 0; i < 6; i++) {
            Serial.print(rc_values[i]);
            Serial.print(" ");
        }
        Serial.println();
        Serial.flush();

        // update motors
        if (batt_low) {
            motors.write_ena(false);
        } else {
            motors.write_ena(query.ena);
        }
        motors.write_vel_left(query.v1);
        motors.write_vel_right(query.v2);

        if (query.active) {
            delay(10);
        } else {
            delay(500);
        }
    }
}


void loop() {
}
