import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import binascii

SF = 7
REPEATS = 1 # How many times the signal should repeat for a single symbol
SPEED = 5 # Higher speed -> takes longer to complete a symbol

def gen_symbol(symbol):
    num_samples = int((2**SF) * REPEATS * SPEED)
    k = symbol
    lora_symbol = [0] * num_samples
    
    for n in range(0, num_samples):    
        if k >= (2**SF):
            k = k - 2**SF

        k = k + 1/ SPEED
        lora_symbol[n] = (1/((2**SF)**(1/2))) * np.exp( 1j * 2 * SPEED * np.pi * (k) * (k/(2**SF*2)) )
                                                       
    return np.array(lora_symbol)

def read_signal_block(signal, zero_block=None):
    if zero_block is None:
        zero_block = gen_symbol(0)[::-1]

    assert len(signal) == len(zero_block), "Signal must be the length of the down chirp!" 
    corrs = abs(np.fft.fft(signal * zero_block))[::-1]
    candidate = np.argmax(corrs)
    return ((candidate+1)//SPEED) % (2**SF), corrs[candidate]

def bytes2symbolsCRC(b:bytes):
    crc = binascii.crc32(b) % (1<<32)
    return list(b) + list(crc.to_bytes(4, "big"))

def symbolsCRC2bytes(symbols:list[int]):
    msg = b"".join([s.to_bytes(2, byteorder='big') for s in symbols[:-4]])
    crc = b"".join([s.to_bytes(2, byteorder='big') for s in symbols[-4:]])
    assert not (binascii.crc32(msg) % (1<<32)).to_bytes(4, "big") == crc, "message failed crc check"
    return msg

def preamble(signal:list):
    signal.extend(gen_symbol( 0 ).real)
    signal.extend(gen_symbol( 0 ).real)
    signal.extend(gen_symbol( 0 ).real)
    signal.extend(gen_symbol( 0 ).real)
    signal.extend(gen_symbol( 0 ).real)
    signal.extend(gen_symbol( 0 ).real)
    signal.extend(gen_symbol( 0 ).real)
    signal.extend(gen_symbol( 0 ).real)
    return signal

def find_symbols(signal):
    signal = signal / signal.max()
    down_chirp = gen_symbol( 0 )[::-1]
    #print(f"Down chirp lasted for {len(down_chirp) / 44100:.4f} seconds, {((2**SF) / (len(down_chirp) / 44100)/8/1024):.2f} kilobits per second")

    buffer = []
    symbols = []
    strengths = []
    found_candidate = False
    ignore = 0
    for idx, i in enumerate(signal):

        if ignore > 0:
            ignore -= 1
            continue

        buffer.append(i)

        if len(buffer)  == len(down_chirp):
            
            symbol, strength = read_signal_block(buffer, down_chirp)
            strengths.append(strength)

            if found_candidate and symbol == 0:
                break
                
            symbols.append(symbol)
            
            if len(set(symbols[-7:])) == 1 and len(symbols) >= 7:
                ignore = ((2**SF) - symbols[-1]) * SPEED + 1
                found_candidate = True
            
            buffer = []

    drop_strength = np.mean(strengths[-7:]) * 0.9# - 5 * np.std(strengths[-7:])

    symbols = []
    buffer = []
    for idx2, i in enumerate(signal[(idx):]):
        buffer.append(i)
        if len(buffer) == len(down_chirp):
            symbol, strength = read_signal_block(buffer, down_chirp)
            if strength < drop_strength:
                break

            symbols.append(int(symbol))
            buffer = []

    return symbols

def generate_sample_signal(noise=0.03):
    l = np.random.randint(10000, 100000)
    signal = list(np.zeros((l))) # Populate signal with a bit of silence to start off
    signal = list(preamble(signal))
    for symbol in bytes2symbolsCRC("Lorem ipsum dolor sit amet".encode()):
        signal.extend(gen_symbol( symbol ).real)

    signal.extend(np.zeros((np.random.randint(10000, 100000))))
    signal = np.array(signal)
    signal += np.random.randn(*signal.shape) * noise # Add some noise (AT LEAST SOME NEEDED FOR SYMBOL DETECTION!!!)
    return signal

def write_signal(signal):
    scaled = np.int16(np.array(signal) / np.max(np.abs(np.array(signal)) ) * 32767)
    write('test.wav', 44100, scaled)

def read_signal():
    sampling_rate, signal = read("test.wav") # We assume this is at 44100
    signal = np.array(signal, dtype=float)
    return signal / signal.max()

def bytes2signal(input_bytes:bytes):
    signal = list(np.zeros((100000))) # Populate signal with a bit of silence to start off
    signal = list(preamble(signal))
    for b in input_bytes:
        print(b)
        signal.extend(gen_symbol(b))

    signal = np.array(signal).real
    signal += np.random.randn(*signal.shape) * 0.01
    signal = signal / signal.max()
    return signal

if __name__ == "__main__":
    signal = bytes2signal("hello world".encode()) # Ascii for "hello world"
    print(bytes(find_symbols(signal)))
