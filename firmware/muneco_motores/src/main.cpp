#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#include <micro_ros_platformio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>

#include <std_msgs/msg/float32_multi_array.h>

#if !defined(MICRO_ROS_TRANSPORT_ARDUINO_SERIAL)
#error This example is only available for Arduino framework with serial transport.
#endif

// ============================================================
// Muneco bipedo - Pata izquierda - Firmware ESP32 con micro-ROS
//
// 3 servos directo a GPIO (PWM LEDC 50 Hz, 16 bits), sin PCA9685:
//   r  Hip Roll   DS3245 (180 grados) -> GPIO27
//   h  Hip Pitch  DS3240 (270 grados) -> GPIO26
//   k  Knee       DS3240 (270 grados) -> GPIO25
//
// Suscripcion:  /servo_commands  (std_msgs/Float32MultiArray)
//   data[i] = q_i + offset_i  (lo que publica teleop_node)
//   El firmware resta el offset para recuperar q_i (grados).
//
// Conversion q -> angulo fisico del servo:
//   angulo = cero + q        (inv = false)
//   angulo = cero - q        (inv = true)
//   angulo 0 -> 500 us, angulo = rango -> 2500 us
//
// Publicacion:  /servo_states  (std_msgs/Float32MultiArray)
//   Posicion MEDIDA por el encoder de cada articulacion, en el
//   mismo marco que /servo_commands (q + offset). Si un encoder
//   no responde o no esta calibrado se publica la posicion de la
//   rampa de ese servo.
//
// Publicacion:  /encoder_raw  (std_msgs/Float32MultiArray)
//   Lectura cruda de los AS5600 (0..4095, -1 si no responde).
//   Solo para calibrar ENC_CERO / ENC_SIGNO.
//
// Encoders AS5600 (0x36) detras de un TCA9548A (0x70), I2C SDA 21 SCL 22:
//   q = signo * (raw - cero) * 360/4096   (diferencia llevada a +-180)
//
// Los servos arrancan apagados (sin pulso). Con el primer comando
// cada servo se activa en q = 0 y desde ahi viaja con
// rampa de velocidad VEL_DEG_S hasta el objetivo.
// ============================================================

const int FREQ = 50;
const int RES  = 16;

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  #define PWM_ATTACH(pin, ch) ledcAttach(pin, FREQ, RES)
  #define PWM_WRITE(pin, ch, duty) ledcWrite(pin, duty)
#else
  #define PWM_ATTACH(pin, ch) do { ledcSetup(ch, FREQ, RES); ledcAttachPin(pin, ch); } while (0)
  #define PWM_WRITE(pin, ch, duty) ledcWrite(ch, duty)
#endif

#define LED_PIN 2
#define NUM_SERVOS 3

struct Servo {
  const char* name;
  int pin, ch;
  float rango;        // recorrido total en grados (180 o 270)
  bool inv;           // invertir sentido de giro
  float offset;       // offset que suma teleop_node (q + offset)
  float cero;         // angulo fisico del servo (0..rango) cuando q = 0  <-- AJUSTABLE
  float actual, target;   // grados (q)
  bool on;
};

// Encoders: canal del TCA9548A, raw cuando q = 0 (-1 = sin calibrar)
// y signo para que +raw siga a +q.                       <-- AJUSTABLE
struct Encoder {
  uint8_t canal;
  int cero;
  int signo;
  int raw;            // ultima lectura (-1 = no responde)
};

Encoder E[NUM_SERVOS] = {
  // canal, cero, signo
  {0, 3035, +1, -1},  // roll cadera     (calibrado: q=0 -> 3035, q=+20 -> 3242)
  {6, 3798, -1, -1},  // pitch cadera    (calibrado: q=0 -> 3798, q=+20 -> 3575)
  {7, 1083, -1, -1},  // pitch rodilla   (calibrado: q=0 -> 1083, q=+20 -> 868)
};

Servo S[NUM_SERVOS] = {
  // name,                 pin, ch, rango, inv,   offset, cero
  {"roll cadera DS3245",   27, 0, 180,  true,  180,    180, 0, 0, false},
  {"pitch cadera DS3240",  26, 1, 270,  false, 135,    135, 0, 0, false},
  {"pitch rodilla DS3240", 25, 2, 270,  true,  135,    135, 0, 0, false},  // inv: +q = rodilla se recoge (como en RViz/IK)
};

const float VEL_DEG_S = 30;   // velocidad de la rampa
unsigned long t0;

// ============================================================
// Conversion grados -> PWM
// ============================================================

uint32_t usToDuty(float us) {
  return (uint32_t)(us / (1000000.0f / FREQ) * 65535.0f);
}

// Limites de q para no salirse del recorrido fisico del servo
float qMin(const Servo& s) { return s.inv ? -(s.rango - s.cero) : -s.cero; }
float qMax(const Servo& s) { return s.inv ?  s.cero : (s.rango - s.cero); }

float degToUs(const Servo& s, float q) {
  float ang = s.cero + (s.inv ? -q : q);          // angulo fisico del servo
  ang = constrain(ang, 0.0f, s.rango);
  return 500.0f + ang * (2000.0f / s.rango);      // 0 -> 500 us, rango -> 2500 us
}

void escribir(Servo& s) {
  PWM_WRITE(s.pin, s.ch, usToDuty(degToUs(s, s.actual)));
}

void activar(Servo& s) {
  if (!s.on) {
    s.on = true;
    s.actual = 0;
    escribir(s);
  }
}

// ============================================================
// Encoders AS5600 detras del TCA9548A
// ============================================================

const int PIN_SDA = 21;
const int PIN_SCL = 22;
const uint8_t ADDR_TCA9548A = 0x70;
const uint8_t ADDR_AS5600   = 0x36;

bool tcaSelect(uint8_t canal) {
  Wire.beginTransmission(ADDR_TCA9548A);
  Wire.write(canal > 7 ? 0 : (1 << canal));
  return Wire.endTransmission() == 0;
}

// Angulo crudo 0..4095, -1 si no responde
int leerAS5600(uint8_t canal) {
  if (!tcaSelect(canal)) return -1;
  Wire.beginTransmission(ADDR_AS5600);
  Wire.write(0x0C);                              // RAW ANGLE
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom((int)ADDR_AS5600, 2) != 2) return -1;
  int hi = Wire.read(), lo = Wire.read();
  return ((hi & 0x0F) << 8) | lo;
}

void leerEncoders() {
  for (int i = 0; i < NUM_SERVOS; i++) E[i].raw = leerAS5600(E[i].canal);
  tcaSelect(8);                                  // ningun canal
}

// q medido (grados); NAN si no hay lectura o no esta calibrado
float qEncoder(const Encoder& e) {
  if (e.raw < 0 || e.cero < 0) return NAN;
  int d = e.raw - e.cero;
  if (d >= 2048) d -= 4096;
  if (d < -2048) d += 4096;
  return e.signo * d * (360.0f / 4096.0f);
}

// ============================================================
// micro-ROS
// ============================================================

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if ((temp_rc != RCL_RET_OK)) { return false; } }
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; (void)temp_rc; }

rclc_support_t support;
rcl_node_t node;
rcl_allocator_t allocator;
rcl_subscription_t subscription;
rcl_publisher_t publisher;
rcl_publisher_t raw_publisher;
rclc_executor_t executor;
rcl_timer_t timer;

std_msgs__msg__Float32MultiArray cmd_msg;
std_msgs__msg__Float32MultiArray state_msg;
std_msgs__msg__Float32MultiArray raw_msg;

float cmd_data[NUM_SERVOS];
float state_data[NUM_SERVOS];
float raw_data[NUM_SERVOS];

enum Estado { ESPERANDO_AGENTE, AGENTE_DISPONIBLE, AGENTE_CONECTADO, AGENTE_DESCONECTADO };
Estado estado = ESPERANDO_AGENTE;

// Recibe q + offset de teleop_node, guarda q como objetivo.
void cmd_callback(const void* msgin) {
  const std_msgs__msg__Float32MultiArray* msg =
      (const std_msgs__msg__Float32MultiArray*)msgin;

  if (msg->data.size < NUM_SERVOS) return;

  for (int i = 0; i < NUM_SERVOS; i++) {
    float q = constrain(msg->data.data[i] - S[i].offset, qMin(S[i]), qMax(S[i]));
    activar(S[i]);
    S[i].target = q;
  }
}

// Publica la posicion medida en el marco de /servo_commands
// (rampa si el encoder no responde o no esta calibrado) y el raw.
void state_timer_callback(rcl_timer_t* timer, int64_t last_call_time) {
  (void)last_call_time;
  if (timer == NULL) return;
  leerEncoders();
  for (int i = 0; i < NUM_SERVOS; i++) {
    float q = qEncoder(E[i]);
    state_data[i] = (isnan(q) ? S[i].actual : q) + S[i].offset;
    raw_data[i] = E[i].raw;
  }
  RCSOFTCHECK(rcl_publish(&publisher, &state_msg, NULL));
  RCSOFTCHECK(rcl_publish(&raw_publisher, &raw_msg, NULL));
}

bool crear_entidades() {
  allocator = rcl_get_default_allocator();

  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "esp32_muneco_motores", "", &support));

  cmd_msg.data.data = cmd_data;
  cmd_msg.data.capacity = NUM_SERVOS;
  cmd_msg.data.size = 0;

  RCCHECK(rclc_subscription_init_default(
      &subscription, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
      "servo_commands"));

  state_msg.data.data = state_data;
  state_msg.data.capacity = NUM_SERVOS;
  state_msg.data.size = NUM_SERVOS;

  RCCHECK(rclc_publisher_init_default(
      &publisher, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
      "servo_states"));

  raw_msg.data.data = raw_data;
  raw_msg.data.capacity = NUM_SERVOS;
  raw_msg.data.size = NUM_SERVOS;

  RCCHECK(rclc_publisher_init_default(
      &raw_publisher, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
      "encoder_raw"));

  RCCHECK(rclc_timer_init_default(
      &timer, &support, RCL_MS_TO_NS(100), state_timer_callback));

  executor = rclc_executor_get_zero_initialized_executor();
  RCCHECK(rclc_executor_init(&executor, &support.context, 2, &allocator));
  RCCHECK(rclc_executor_add_subscription(
      &executor, &subscription, &cmd_msg, &cmd_callback, ON_NEW_DATA));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  return true;
}

void destruir_entidades() {
  rmw_context_t* rmw_context = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_context, 0);

  RCSOFTCHECK(rcl_publisher_fini(&publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&raw_publisher, &node));
  RCSOFTCHECK(rcl_subscription_fini(&subscription, &node));
  RCSOFTCHECK(rcl_timer_fini(&timer));
  RCSOFTCHECK(rclc_executor_fini(&executor));
  RCSOFTCHECK(rcl_node_fini(&node));
  RCSOFTCHECK(rclc_support_fini(&support));
}

// ============================================================
// Setup / Loop
// ============================================================

void setup() {
  pinMode(LED_PIN, OUTPUT);

  // Servos apagados (sin pulso) hasta el primer comando
  for (int i = 0; i < NUM_SERVOS; i++) {
    PWM_ATTACH(S[i].pin, S[i].ch);
    PWM_WRITE(S[i].pin, S[i].ch, 0);
  }

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  Wire.setTimeOut(5);

  Serial.begin(115200);
  set_microros_serial_transports(Serial);
  delay(500);

  t0 = millis();
}

// Ejecuta `X` cada `MS` milisegundos (patron de los ejemplos de micro-ROS)
#define CADA(MS, X) do { static int64_t t_ = -1; if (t_ == -1) t_ = uxr_millis(); \
                         if ((int32_t)(uxr_millis() - t_) > (MS)) { X; t_ = uxr_millis(); } } while (0)

void loop() {
  // --- Conexion con el agente (se reconecta solo si se cae) ---
  switch (estado) {
    case ESPERANDO_AGENTE:
      CADA(500, estado = (RMW_RET_OK == rmw_uros_ping_agent(100, 1)) ? AGENTE_DISPONIBLE : ESPERANDO_AGENTE;);
      digitalWrite(LED_PIN, (millis() / 250) % 2);   // parpadeo: esperando agente
      break;
    case AGENTE_DISPONIBLE:
      estado = crear_entidades() ? AGENTE_CONECTADO : ESPERANDO_AGENTE;
      if (estado == ESPERANDO_AGENTE) destruir_entidades();
      break;
    case AGENTE_CONECTADO:
      CADA(500, estado = (RMW_RET_OK == rmw_uros_ping_agent(100, 1)) ? AGENTE_CONECTADO : AGENTE_DESCONECTADO;);
      if (estado == AGENTE_CONECTADO) rclc_executor_spin_some(&executor, RCL_MS_TO_NS(5));
      digitalWrite(LED_PIN, HIGH);                    // fijo: conectado
      break;
    case AGENTE_DESCONECTADO:
      destruir_entidades();
      estado = ESPERANDO_AGENTE;
      break;
  }

  // --- Rampa de velocidad: los servos mantienen posicion aunque se caiga el agente ---
  unsigned long ahora = millis();
  if (ahora - t0 >= 20) {
    float dt = (ahora - t0) / 1000.0f;
    t0 = ahora;
    float paso = VEL_DEG_S * dt;
    for (int i = 0; i < NUM_SERVOS; i++) {
      Servo& s = S[i];
      if (!s.on) continue;
      if (fabs(s.target - s.actual) <= paso) s.actual = s.target;
      else s.actual += (s.target > s.actual) ? paso : -paso;
      escribir(s);
    }
  }
}