import cryptomath

# Các hằng số
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def encrypt_message(message, key_a, key_b):
    # Kiểm tra Key A có hợp lệ không (phải nguyên tố cùng nhau với 26)
    if cryptomath.gcd(key_a, len(SYMBOLS)) != 1:
        return 'Key A và bộ chữ cái không nguyên tố cùng nhau. Hãy chọn Key A khác!'

    ciphertext = ''
    for symbol in message.upper():
        if symbol in SYMBOLS:
            # Tìm chỉ số của ký tự
            sym_index = SYMBOLS.find(symbol)
            # Áp dụng công thức: (index * KeyA + KeyB) % 26
            new_index = (sym_index * key_a + key_b) % len(SYMBOLS)
            ciphertext += SYMBOLS[new_index]
        else:
            ciphertext += symbol # Giữ nguyên dấu cách/số
            
    return ciphertext

def decrypt_message(ciphertext, key_a, key_b):
    # Tìm nghịch đảo modulo của Key A
    mod_inverse_a = cryptomath.findModInverse(key_a, len(SYMBOLS))
    
    if mod_inverse_a is None:
        return 'Không thể giải mã với Key A này!'

    plaintext = ''
    for symbol in ciphertext.upper():
        if symbol in SYMBOLS:
            sym_index = SYMBOLS.find(symbol)
            # Công thức giải mã: (index - KeyB) * modInverseA % 26
            new_index = (sym_index - key_b) * mod_inverse_a % len(SYMBOLS)
            plaintext += SYMBOLS[new_index]
        else:
            plaintext += symbol
            
    return plaintext

# --- CHẠY THỬ ---
my_msg = "HELLO CRYPTOGRAPHY"
ka, kb = 7, 10  # Hai khóa bất kỳ (ka phải lẻ và không phải 13)

encrypted = encrypt_message(my_msg, ka, kb)
decrypted = decrypt_message(encrypted, ka, kb)

print(f"Gốc: {my_msg}")
print(f"Mã hóa: {encrypted}")
print(f"Giải mã: {decrypted}")