const int PIN_STBY = 6;
const int PIN_AIN1 = 8;
const int PIN_AIN2 = 7;
const int PIN_PWM  = 9;

volatile long pulsos_total = 0;
long pulsos_anterior = 0;
float caudal_mlmin = 0;

void contar_pulso() { pulsos_total++; }

// Lectura atomica del contador: en AVR un long son 4 bytes y la ISR puede
// interrumpir a mitad de lectura, corrompiendo el valor.
long leer_pulsos() {
  noInterrupts();
  long pulsos = pulsos_total;
  interrupts();
  return pulsos;
}

const int PWM_MIN = 122;
const int PWM_MAX = 138;
const float CAUDAL_MIN = 120.0;
const float CAUDAL_MAX = 150.0;

// START: rampa lineal de PWM_LIMPIEZA (max) a PWM_MIN en RAMPA_MS,
// luego entra el fuzzy.
const unsigned long RAMPA_MS = 8000;
// STOP: desde el PWM actual del fuzzy sube PASO_PWM_STOP cada
// PASO_MS_STOP hasta PWM_STOP_TARGET (purga), y ahi se apaga. Con el
// rango fuzzy (~130) son ~4 escalones de 3 s hasta 255, es decir ~15 s
// antes de apagar del todo.
const int PASO_PWM_STOP = 40;
const unsigned long PASO_MS_STOP = 3000UL;
const int PWM_LIMPIEZA = 255;
const int PWM_STOP_TARGET = 255;

const int VENTANA = 10;
float historial[VENTANA];
int idx = 0;
bool ventana_llena = false;

enum Estado { OFF, LIMPIEZA, RAMPA_SUBIDA, FUZZY_ACTIVO, RAMPA_STOP };
Estado estado = OFF;

// Nombres de membresia en una sola fuente de verdad (evita literales
// sueltos repetidos) y sin String, que fragmenta el heap del AVR en
// corridas largas.
enum Membresia { MEM_MUY_POCAS, MEM_POCAS, MEM_MEDIA, MEM_MUCHAS, MEM_NINGUNA };
const char* const NOMBRES_MEMBRESIA[] = {"MUY_POCAS", "POCAS", "MEDIA", "MUCHAS", "-"};

// Prototipos manuales: el IDE de Arduino genera los prototipos automaticos
// antes de la definicion de estos enums, lo que rompe la compilacion.
void iniciar_rampa(int desde, int hasta, Estado siguiente);
Membresia membresia_dominante(float x);

int pwm_actual = 0;
unsigned long ramp_start_ms = 0;
int ramp_pwm_desde = 0;
int ramp_pwm_hasta = 0;
Membresia membresia_actual = MEM_NINGUNA;

// ---- Funciones de membresia (variable de entrada: conteo de particulas) ----

float muy_pocas(float x) {
  if (x <= 0) return 1.0;
  if (x >= 5) return 0.0;
  return (5.0 - x) / 5.0;
}

float pocas(float x) {
  if (x <= 0 || x >= 10) return 0.0;
  if (x <= 5) return x / 5.0;
  return (10.0 - x) / 5.0;
}

float media(float x) {
  if (x <= 5 || x >= 20) return 0.0;
  if (x <= 12) return (x - 5.0) / 7.0;
  return (20.0 - x) / 8.0;
}

float muchas(float x) {
  if (x <= 15) return 0.0;
  if (x >= 25) return 1.0;
  return (x - 15.0) / 10.0;
}

Membresia membresia_dominante(float x) {
  float u1 = muy_pocas(x);
  float u2 = pocas(x);
  float u3 = media(x);
  float u4 = muchas(x);
  float max_u = max(max(u1, u2), max(u3, u4));
  if (max_u == u1) return MEM_MUY_POCAS;
  if (max_u == u2) return MEM_POCAS;
  if (max_u == u3) return MEM_MEDIA;
  return MEM_MUCHAS;
}

int fuzzy_pwm(float particulas) {
  float u1 = muy_pocas(particulas);
  float u2 = pocas(particulas);
  float u3 = media(particulas);
  float u4 = muchas(particulas);
  float num = u1 * 138 + u2 * 133 + u3 * 128 + u4 * 122;
  float den = u1 + u2 + u3 + u4;
  if (den == 0) return PWM_MAX;
  return constrain((int)(num / den), PWM_MIN, PWM_MAX);
}

float promedio(float nuevo_valor) {
  historial[idx] = nuevo_valor;
  idx = (idx + 1) % VENTANA;
  if (idx == 0) ventana_llena = true;
  int n = ventana_llena ? VENTANA : idx;
  float suma = 0;
  for (int i = 0; i < n; i++) suma += historial[i];
  return suma / n;
}

struct PuntoCal { int pwm; float caudal; };
const PuntoCal TABLA_CAL[] = {
  {120, 118}, {133, 130}, {140, 156}, {147, 171}, {150, 176},
  {155, 188}, {160, 198}, {163, 204}, {180, 235}, {210, 303}, {255, 390}
};
const int N_CAL = sizeof(TABLA_CAL) / sizeof(TABLA_CAL[0]);

int caudal_a_pwm(float caudal) {
  caudal = constrain(caudal, CAUDAL_MIN, CAUDAL_MAX);
  for (int i = 0; i < N_CAL - 1; i++) {
    if (caudal <= TABLA_CAL[i + 1].caudal) {
      float c0 = TABLA_CAL[i].caudal, c1 = TABLA_CAL[i + 1].caudal;
      int p0 = TABLA_CAL[i].pwm, p1 = TABLA_CAL[i + 1].pwm;
      return (int)(p0 + (caudal - c0) * (p1 - p0) / (c1 - c0));
    }
  }
  return TABLA_CAL[N_CAL - 1].pwm;
}

// Interpolacion de la curva PWM -> caudal levantada por el metodo
// volumetrico (seccion 2 del documento de calibracion). Solo informativo.
float interpolar_caudal(int pwm) {
  if (pwm <= 120) return 118;
  else if (pwm <= 150) return 118 + (pwm - 120) * (176 - 118.0) / (150 - 120);
  else if (pwm <= 180) return 176 + (pwm - 150) * (235 - 176.0) / (180 - 150);
  else if (pwm <= 210) return 235 + (pwm - 180) * (303 - 235.0) / (210 - 180);
  else return 303 + (pwm - 210) * (390 - 303.0) / (255 - 210);
}

// ---- Motor ----

void set_pwm(int pwm) {
  pwm = constrain(pwm, 0, 255);
  digitalWrite(PIN_AIN1, LOW);
  digitalWrite(PIN_AIN2, HIGH);
  digitalWrite(PIN_STBY, HIGH);
  analogWrite(PIN_PWM, pwm);
  pwm_actual = pwm;
}

const char* estado_str() {
  switch (estado) {
    case OFF: return "OFF";
    case LIMPIEZA: return "LIMPIEZA";
    case RAMPA_SUBIDA: return "RAMPA_SUBIDA";
    case FUZZY_ACTIVO: return "FUZZY_ACTIVO";
    case RAMPA_STOP: return "RAMPA_STOP";
  }
  return "OFF";
}

void iniciar_rampa(int desde, int hasta, Estado siguiente) {
  ramp_start_ms = millis();
  ramp_pwm_desde = desde;
  ramp_pwm_hasta = hasta;
  estado = siguiente;
}

void imprimir_detenido() {
  float volumen_ml = leer_pulsos() / 4.85;
  float volumen_litros = volumen_ml / 1000.0;
  Serial.print("SISTEMA_DETENIDO");
  Serial.print(",VOL="); Serial.print(volumen_ml, 1);
  Serial.print(",LIT="); Serial.println(volumen_litros, 4);
}

// Recalcula el PWM de la rampa en curso a partir de millis(): nada de
// delay(), asi que no bloquea la lectura del puerto serie.
void actualizar_rampa() {
  unsigned long transcurrido = millis() - ramp_start_ms;

  if (estado == RAMPA_SUBIDA) {
    // Arranque: lineal de PWM_LIMPIEZA (max) a PWM_MIN en RAMPA_MS.
    if (transcurrido >= RAMPA_MS) {
      set_pwm(ramp_pwm_hasta);
      estado = FUZZY_ACTIVO;
      return;
    }
    int pwm = ramp_pwm_desde +
              (long)(ramp_pwm_hasta - ramp_pwm_desde) * transcurrido / RAMPA_MS;
    set_pwm(pwm);
  }

  else if (estado == RAMPA_STOP) {
    // Parada con purga: desde el PWM donde quedo el fuzzy sube
    // PASO_PWM_STOP cada PASO_MS_STOP hasta ramp_pwm_hasta; el escalon
    // final tambien se mantiene su intervalo antes de apagar del todo.
    if (transcurrido < PASO_MS_STOP) return;
    if (pwm_actual >= ramp_pwm_hasta) {
      analogWrite(PIN_PWM, 0);
      pwm_actual = 0;
      estado = OFF;
      imprimir_detenido();
      return;
    }
    set_pwm(min(pwm_actual + PASO_PWM_STOP, ramp_pwm_hasta));
    ramp_start_ms = millis();
  }
}

void setup() {
  pinMode(PIN_STBY, OUTPUT);
  pinMode(PIN_AIN1, OUTPUT);
  pinMode(PIN_AIN2, OUTPUT);
  pinMode(PIN_PWM, OUTPUT);
  digitalWrite(PIN_STBY, HIGH);
  // AIN1=LOW, AIN2=HIGH invierte el sentido del motor para que la bomba
  // succione agua del deposito en lugar de empujarla.
  digitalWrite(PIN_AIN1, LOW);
  digitalWrite(PIN_AIN2, HIGH);
  analogWrite(PIN_PWM, 0);

  pinMode(2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(2), contar_pulso, RISING);

  Serial.begin(9600);
  // readStringUntil bloquea hasta su timeout si no llega el '\n'; con el
  // default (1000 ms) una linea incompleta congelaria la rampa 1 segundo.
  Serial.setTimeout(50);
  Serial.println("ESCLAVO_LISTO");
  Serial.println("Comandos: CALIBRATE | START | STOP | N<conteo_frame> | c<caudal>");
}

void loop() {
  actualizar_rampa();

  static unsigned long ultimo = 0;
  if (millis() - ultimo >= 1000) {
    ultimo = millis();
    long pulsos_ahora = leer_pulsos();
    float pulsos_seg = pulsos_ahora - pulsos_anterior;
    pulsos_anterior = pulsos_ahora;
    caudal_mlmin = (pulsos_seg / 4.85) * 60.0;

    float volumen_ml = pulsos_ahora / 4.85;
    Serial.print("ST="); Serial.print(estado_str());
    Serial.print(",PWM="); Serial.print(pwm_actual);
    Serial.print(",CS="); Serial.print(caudal_mlmin, 1);
    Serial.print(",MEM="); Serial.print(NOMBRES_MEMBRESIA[membresia_actual]);
    Serial.print(",VOL="); Serial.println(volumen_ml, 1);
  }

  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();

    if (comando == "CALIBRATE") {
      set_pwm(PWM_LIMPIEZA);
      estado = LIMPIEZA;
    }

    else if (comando == "START") {
      noInterrupts();
      pulsos_total = 0;
      interrupts();
      // Sin esto, el primer CS tras el reset del contador sale negativo
      // (pulsos_anterior quedaria con el valor de la corrida anterior).
      pulsos_anterior = 0;
      ventana_llena = false;
      idx = 0;
      membresia_actual = MEM_NINGUNA;
      set_pwm(PWM_LIMPIEZA);
      iniciar_rampa(PWM_LIMPIEZA, PWM_MIN, RAMPA_SUBIDA);
    }

    else if (comando == "STOP") {
      if (estado != OFF) {
        // Purga de parada: desde el PWM actual sube en escalones hasta
        // PWM_STOP_TARGET y recien ahi se apaga (ver actualizar_rampa).
        iniciar_rampa(pwm_actual, PWM_STOP_TARGET, RAMPA_STOP);
      }
    }

    // Conteo de microplasticos del frame actual, enviado por el maestro
    // (Raspberry Pi). Alimenta la logica difusa que fija el PWM. Solo
    // tiene efecto con el fuzzy activo: durante las rampas se ignora.
    else if (comando.startsWith("N")) {
      if (estado != FUZZY_ACTIVO) {
        Serial.println("ERROR:en_rampa");
        return;
      }
      float particulas = comando.substring(1).toFloat();
      if (particulas < 0 || particulas > 100) {
        Serial.println("ERROR:valor_invalido");
        return;
      }
      float prom = promedio(particulas);
      int pwm = fuzzy_pwm(prom);
      set_pwm(pwm);
      float caudal_fuzzy = interpolar_caudal(pwm);
      membresia_actual = membresia_dominante(prom);
      Serial.print("OK:");
      Serial.print("P="); Serial.print(particulas);
      Serial.print(",PROM="); Serial.print(prom, 2);
      Serial.print(",PWM="); Serial.print(pwm);
      Serial.print(",CF="); Serial.print(caudal_fuzzy, 1);
      Serial.print(",CS="); Serial.print(caudal_mlmin, 1);
      Serial.print(",MEM="); Serial.println(NOMBRES_MEMBRESIA[membresia_actual]);
    }

    // Override manual de caudal (botones +/- del dashboard). Solo para
    // pruebas: fuerza el PWM directamente, ignorando la logica difusa
    // hasta que llegue el siguiente conteo 'N'. Se retirara en produccion.
    else if (comando.startsWith("c")) {
      if (estado != FUZZY_ACTIVO) {
        Serial.println("ERROR:en_rampa");
        return;
      }
      float caudal_deseado = comando.substring(1).toFloat();
      if (caudal_deseado < CAUDAL_MIN || caudal_deseado > CAUDAL_MAX) {
        Serial.println("ERROR:rango 120-150 ml/min");
        return;
      }
      int pwm = caudal_a_pwm(caudal_deseado);
      set_pwm(pwm);
      float volumen_ml = leer_pulsos() / 4.85;
      float volumen_litros = volumen_ml / 1000.0;
      Serial.print("OK:");
      Serial.print("CF="); Serial.print(caudal_deseado, 1);
      Serial.print(",PWM="); Serial.print(pwm);
      Serial.print(",VOL="); Serial.print(volumen_ml, 1);
      Serial.print(",LIT="); Serial.println(volumen_litros, 4);
    }
  }
}
