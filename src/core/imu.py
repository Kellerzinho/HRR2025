# src/core/imu.py
import sys
import time
import math

# Importa smbus ou smbus2 dependendo do sistema operacional
if sys.platform.startswith("win"):
    try:
        from smbus2 import SMBus
    except ImportError:
        # Se não estiver disponível, define uma classe dummy para testes
        class SMBus:
            def __init__(self, *args, **kwargs): pass
else:
    try:
        from smbus import SMBus
    except ImportError:
        class SMBus:
            def __init__(self, *args, **kwargs): pass

class IMUReader:
    def __init__(self, mpu_address=0x68, ak_address=0x0C, bus_id=1):
        """
        Inicializa a comunicação I²C com o MPU9250 e o magnetômetro AK8963.
        :param mpu_address: Endereço I²C do MPU9250 (normalmente 0x68)
        :param ak_address: Endereço I²C do AK8963 (normalmente 0x0C)
        :param bus_id: Número do barramento I²C (geralmente 1)
        """
        self.mpu_address = mpu_address
        self.ak_address = ak_address
        self.bus = SMBus(bus_id)
        self._initialize_sensor()

    def _initialize_sensor(self):
        try:
            # Despertar o MPU9250 (escreve 0 no registrador de gerenciamento de energia)
            self.bus.write_byte_data(self.mpu_address, 0x6B, 0)
            time.sleep(0.1)
        except Exception as e:
            print(f"[IMU] Erro ao despertar MPU9250: {e}")
        
        try:
            # Ativar o modo bypass para acessar diretamente o AK8963.
            # Registrador 0x37 (INT_PIN_CFG): configurar bit 1 para 1.
            self.bus.write_byte_data(self.mpu_address, 0x37, 0x02)
            time.sleep(0.1)
        except Exception as e:
            print(f"[IMU] Erro ao configurar bypass: {e}")
        
        try:
            # Configurar o AK8963: escreva em CNTL1 (registrador 0x0A) para:
            # - Operar em modo contínuo de medição 2 (100Hz) com saída de 16 bits.
            self.bus.write_byte_data(self.ak_address, 0x0A, 0x16)
            time.sleep(0.1)
        except Exception as e:
            print(f"[IMU] Erro ao configurar o AK8963: {e}")
        
        print("[IMU] MPU9250 e AK8963 inicializados com sucesso (ou com erros tratados).")

    def read_raw_data(self, reg):
        """
        Lê dois bytes do registrador e os combina em um valor de 16 bits.
        Se ocorrer algum erro na leitura, retorna 0.
        """
        try:
            high = self.bus.read_byte_data(self.mpu_address, reg)
            low = self.bus.read_byte_data(self.mpu_address, reg + 1)
            value = (high << 8) | low
            if value > 32767:
                value -= 65536
            return value
        except Exception as e:
            print(f"[IMU] Erro ao ler o registrador {reg}: {e}")
            return 0

    def get_imu_data(self):
        """
        Lê e converte os dados do acelerômetro, giroscópio e magnetômetro.
        Para o MPU9250:
          - Acelerômetro e giroscópio são lidos a partir dos registradores do MPU9250.
          - O magnetômetro é lido do AK8963.
        Retorna um dicionário com:
            - 'timestamp': tempo atual (float)
            - 'accel': (ax, ay, az) em g
            - 'gyro': (gx, gy, gz) em rad/s
            - 'mag': (mx, my, mz) (valores brutos; aplicar escala se necessário)
        """
        # Leitura do acelerômetro (registradores 0x3B a 0x40)
        ax = self.read_raw_data(0x3B)
        ay = self.read_raw_data(0x3D)
        az = self.read_raw_data(0x3F)
        
        # Leitura do giroscópio (registradores 0x43 a 0x48)
        gx = self.read_raw_data(0x43)
        gy = self.read_raw_data(0x45)
        gz = self.read_raw_data(0x47)
        
        # Conversão:
        # Para acelerômetro: sensibilidade de 16384 LSB/g para +/- 2g
        ax = ax / 16384.0
        ay = ay / 16384.0
        az = az / 16384.0
        
        # Para giroscópio: sensibilidade de 131 LSB/(deg/s) para +/- 250 deg/s,
        # converte para rad/s
        gx = math.radians(gx / 131.0)
        gy = math.radians(gy / 131.0)
        gz = math.radians(gz / 131.0)
        
        # Leitura do magnetômetro (AK8963):
        try:
            st1 = self.bus.read_byte_data(self.ak_address, 0x02)
        except Exception as e:
            print(f"[IMU] Erro ao ler ST1 do AK8963: {e}")
            st1 = 0
        
        if st1 & 0x01:
            try:
                data = self.bus.read_i2c_block_data(self.ak_address, 0x03, 7)
                mx = (data[1] << 8) | data[0]
                my = (data[3] << 8) | data[2]
                mz = (data[5] << 8) | data[4]
                if mx > 32767:
                    mx -= 65536
                if my > 32767:
                    my -= 65536
                if mz > 32767:
                    mz -= 65536
            except Exception as e:
                print(f"[IMU] Erro ao ler magnetômetro: {e}")
                mx, my, mz = (0.0, 0.0, 0.0)
        else:
            mx, my, mz = (0.0, 0.0, 0.0)
        
        return {
            'timestamp': time.time(),
            'accel': (ax, ay, az),
            'gyro': (gx, gy, gz),
            'mag': (mx, my, mz)
        }
