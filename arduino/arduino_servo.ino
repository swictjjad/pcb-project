/*
 * PCB缺陷检测与智能分拣系统 — Arduino控制程序
 * 2026 睿抗机器人开发者大赛 · CAIR强体赛道 · AI视觉应用
 * 
 * 功能：
 * 1. 接收串口命令控制舵机角度（合格/不合格分拣）
 * 2. 控制步进电机驱动传送带
 * 3. 读取红外传感器状态（PCB到位检测）
 * 4. 发送传感器触发信号给上位机
 * 
 * 硬件连接：
 * - 舵机: D9 (PWM)
 * - 红外传感器: D2 (数字输入，带中断)
 * - 步进电机驱动: D5(方向), D6(步进)
 * - LED指示灯: D13(内置), D11(合格绿灯), D12(不合格红灯)
 * - 蜂鸣器: D10
 */

#include <Servo.h>

// ============== 引脚定义 ==============
const int SERVO_PIN = 9;           // 舵机PWM引脚
const int SENSOR_PIN = 2;          // 红外传感器（使用中断）
const int MOTOR_DIR_PIN = 5;       // 步进电机方向
const int MOTOR_STEP_PIN = 6;      // 步进电机步进
const int LED_OK_PIN = 11;         // 合格指示灯（绿色）
const int LED_NG_PIN = 12;         // 不合格指示灯（红色）
const int BUZZER_PIN = 10;         // 蜂鸣器
const int LED_BUILTIN_PIN = 13;    // 内置LED

// ============== 参数配置 ==============
const int SERVO_ANGLE_OK = 45;     // 合格品舵机角度
const int SERVO_ANGLE_NG = 135;    // 不合格品舵机角度
const int SERVO_ANGLE_RESET = 90;  // 舵机复位角度
const int SERVO_DELAY = 500;       // 舵机动作保持时间(ms)
const int MOTOR_SPEED = 100;       // 传送带速度（步进脉冲间隔，越小越快）
const int DEBOUNCE_MS = 50;        // 传感器消抖时间(ms)

// ============== 全局变量 ==============
Servo servoMotor;
volatile bool sensorTriggered = false;
unsigned long lastTriggerTime = 0;
bool motorRunning = false;

// ============== 初始化 ==============
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }  // 等待串口连接（Leonardo等需要）
  
  // 引脚初始化
  pinMode(SENSOR_PIN, INPUT_PULLUP);
  pinMode(MOTOR_DIR_PIN, OUTPUT);
  pinMode(MOTOR_STEP_PIN, OUTPUT);
  pinMode(LED_OK_PIN, OUTPUT);
  pinMode(LED_NG_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_BUILTIN_PIN, OUTPUT);
  
  // 舵机初始化
  servoMotor.attach(SERVO_PIN);
  servoMotor.write(SERVO_ANGLE_RESET);
  
  // 中断配置（红外传感器，下降沿触发）
  attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), onSensorTrigger, FALLING);
  
  // 初始状态
  digitalWrite(LED_OK_PIN, LOW);
  digitalWrite(LED_NG_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_BUILTIN_PIN, LOW);
  
  Serial.println("Arduino Ready");
  Serial.println("CMD: SOK=合格, SNG=不合格, SRS=复位, MST=启动, MSP=停止");
}

// ============== 中断服务程序 ==============
void onSensorTrigger() {
  unsigned long now = millis();
  if (now - lastTriggerTime > DEBOUNCE_MS) {
    sensorTriggered = true;
    lastTriggerTime = now;
  }
}

// ============== 主循环 ==============
void loop() {
  // 处理传感器触发
  if (sensorTriggered) {
    sensorTriggered = false;
    Serial.println("TRIGGER:PCB_ARRIVED");
    digitalWrite(LED_BUILTIN_PIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN_PIN, LOW);
  }
  
  // 处理串口命令
  processSerialCommand();
  
  // 运行传送带
  if (motorRunning) {
    stepMotor();
  }
}

// ============== 串口命令处理 ==============
void processSerialCommand() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd == "SOK") {
      // 舵机转到合格位置
      servoMotor.write(SERVO_ANGLE_OK);
      digitalWrite(LED_OK_PIN, HIGH);
      digitalWrite(LED_NG_PIN, LOW);
      tone(BUZZER_PIN, 1000, 200);  // 蜂鸣器短鸣
      Serial.println("OK:SERVO_OK");
      
    } else if (cmd == "SNG") {
      // 舵机转到不合格位置
      servoMotor.write(SERVO_ANGLE_NG);
      digitalWrite(LED_OK_PIN, LOW);
      digitalWrite(LED_NG_PIN, HIGH);
      tone(BUZZER_PIN, 500, 500);   // 蜂鸣器长鸣
      Serial.println("OK:SERVO_NG");
      
    } else if (cmd == "SRS") {
      // 舵机复位
      servoMotor.write(SERVO_ANGLE_RESET);
      digitalWrite(LED_OK_PIN, LOW);
      digitalWrite(LED_NG_PIN, LOW);
      Serial.println("OK:SERVO_RESET");
      
    } else if (cmd == "MST") {
      // 启动传送带
      motorRunning = true;
      digitalWrite(MOTOR_DIR_PIN, HIGH);  // 正转
      Serial.println("OK:MOTOR_START");
      
    } else if (cmd == "MSP") {
      // 停止传送带
      motorRunning = false;
      Serial.println("OK:MOTOR_STOP");
      
    } else if (cmd == "READ") {
      // 读取传感器状态
      int sensorState = digitalRead(SENSOR_PIN);
      Serial.print("SENSOR:");
      Serial.println(sensorState == LOW ? "TRIGGERED" : "CLEAR");
      
    } else {
      Serial.println("ERR:UNKNOWN_CMD");
    }
  }
}

// ============== 步进电机驱动 ==============
void stepMotor() {
  digitalWrite(MOTOR_STEP_PIN, HIGH);
  delayMicroseconds(MOTOR_SPEED);
  digitalWrite(MOTOR_STEP_PIN, LOW);
  delayMicroseconds(MOTOR_SPEED);
}

// ============== 辅助函数 ==============
// 连续分拣动作（合格品）
void sortOK() {
  servoMotor.write(SERVO_ANGLE_OK);
  digitalWrite(LED_OK_PIN, HIGH);
  tone(BUZZER_PIN, 1000, 200);
  delay(SERVO_DELAY);
  servoMotor.write(SERVO_ANGLE_RESET);
  digitalWrite(LED_OK_PIN, LOW);
}

// 连续分拣动作（不合格品）
void sortNG() {
  servoMotor.write(SERVO_ANGLE_NG);
  digitalWrite(LED_NG_PIN, HIGH);
  tone(BUZZER_PIN, 500, 500);
  delay(SERVO_DELAY);
  servoMotor.write(SERVO_ANGLE_RESET);
  digitalWrite(LED_NG_PIN, LOW);
}
