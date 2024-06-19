//#include <IBusBM.h>


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
        analogWrite(3, pwm);
    }

    void write_pwm_right(int pwm) {
        analogWrite(6, pwm);
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
        write_dir_right(vel >= 0);
        write_pwm_right(abs(vel));
    }

    void write_vel_both(int vel) {
        write_vel_left(vel);
        write_vel_right(vel);
    }
};


Motors motors;
//IBusBM ibus;


void setup() {
    delay(100);

    Serial.begin(9600);
    //ibus.begin(Serial1, 2);

    int i = 0;
    motors.write_ena(true);
    while (true) {
        // read RC
        /*
        uint16_t rx_steering = ibus.readChannel(0);
        uint16_t rx_throttle = ibus.readChannel(2);
        uint16_t rx_enable = ibus.readChannel(4);

        int steer = map(rx_steering, 1000, 2000, -50, 50);
        int speed = map(rx_throttle, 1000, 2000, 0, 255);
        int speed1 = speed + steer;
        int speed2 = speed - steer;
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

        delay(10);
        */


        //motors.write_vel_both(100);
        //delay(1000);
        //continue;
        motors.write_ena(true);
        delay(1000);
        motors.write_ena(false);
        delay(1000);
        continue;

        while (i < 255) {
            motors.write_vel_both(i);
            delay(30);
            i++;
        }
        while (i > -255) {
            motors.write_vel_both(i);
            delay(30);
            i--;
        }
    };
}


void loop() {
}
