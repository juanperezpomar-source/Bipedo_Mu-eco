/*
  PRUEBA DE ENCODERS - muñeco
  ---------------------------
  Solo lee: no genera PWM en los servos.
  1. Escanea el bus I2C (debe aparecer el TCA9548A en 0x70).
  2. Por cada canal del TCA lista que hay conectado (debe aparecer 0x36).
  3. Cada 200 ms imprime de los 3 AS5600: estado del iman y angulo crudo.

  Canales: cadera roll = 0, cadera pitch = 6, rodilla pitch = 7.
  I2C: SDA = 21, SCL = 22 (igual que el parcial).
*/
#include <Arduino.h>
#include <Wire.h>

static const int PIN_SDA = 21;
static const int PIN_SCL = 22;
static const uint8_t ADDR_TCA9548A = 0x70;
static const uint8_t ADDR_AS5600   = 0x36;

// Registros AS5600
static const uint8_t REG_STATUS    = 0x0B;
static const uint8_t REG_RAW_ANGLE = 0x0C;
static const uint8_t REG_AGC       = 0x1A;
static const uint8_t REG_MAGNITUDE = 0x1B;

struct Encoder { const char *nombre; uint8_t canal; };
static const Encoder ENCODERS[] = {
  {"cadera roll  ", 0},
  {"cadera pitch ", 6},
  {"rodilla pitch", 7},
};

bool existe(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

bool tcaSelect(uint8_t canal) {
  Wire.beginTransmission(ADDR_TCA9548A);
  Wire.write(canal > 7 ? 0 : (1 << canal));
  return Wire.endTransmission() == 0;
}

bool leer(uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(ADDR_AS5600);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)ADDR_AS5600, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

void escanear(const char *donde) {
  Serial.printf("  %s:", donde);
  int n = 0;
  for (uint8_t a = 1; a < 127; a++) {
    if (existe(a)) { Serial.printf(" 0x%02X", a); n++; }
  }
  if (n == 0) Serial.print(" (nada)");
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(100000);

  Serial.println("\n===== PRUEBA ENCODERS MUNECO =====");
  Serial.printf("I2C SDA=%d SCL=%d\n", PIN_SDA, PIN_SCL);
  if (!existe(ADDR_TCA9548A)) {
    Serial.println("ERROR: TCA9548A no responde en 0x70.");
    escanear("bus principal");
  } else {
    Serial.println("TCA9548A OK en 0x70");
    for (uint8_t c = 0; c < 8; c++) {
      tcaSelect(c);
      char t[16];
      snprintf(t, sizeof(t), "canal %u", c);
      escanear(t);
    }
  }
  tcaSelect(8);  // ningun canal
  Serial.println("==================================\n");
}

void loop() {
  for (const Encoder &e : ENCODERS) {
    uint8_t st, ang[2], agc, mag[2];
    if (!tcaSelect(e.canal) || !leer(REG_STATUS, &st, 1)) {
      Serial.printf("%s (ch %u): NO RESPONDE | ", e.nombre, e.canal);
      continue;
    }
    leer(REG_RAW_ANGLE, ang, 2);
    leer(REG_AGC, &agc, 1);
    leer(REG_MAGNITUDE, mag, 2);
    uint16_t raw = ((ang[0] & 0x0F) << 8) | ang[1];
    const char *iman = !(st & 0x20) ? "SIN IMAN"
                     : (st & 0x08) ? "MUY FUERTE"
                     : (st & 0x10) ? "MUY DEBIL" : "ok";
    Serial.printf("%s (ch %u): raw %4u = %6.2f deg iman %-10s agc %3u | ",
                  e.nombre, e.canal, raw, raw * 360.0f / 4096.0f, iman, agc);
  }
  Serial.println();
  tcaSelect(8);
  delay(200);
}
