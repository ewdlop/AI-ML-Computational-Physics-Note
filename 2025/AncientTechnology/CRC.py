def xor(a, b):
    """Perform XOR operation."""
    result = []
    for i in range(1, len(b)):
        result.append('0' if a[i] == b[i] else '1')
    return ''.join(result)

def mod2_division(data, generator):
    """Perform modulo-2 division for CRC."""
    pick = len(generator)
    temp = data[:pick]

    while pick < len(data):
        if temp[0] == '1':
            temp = xor(generator, temp) + data[pick]
        else:
            temp = xor('0' * pick, temp) + data[pick]
        pick += 1

    # Final step
    if temp[0] == '1':
        temp = xor(generator, temp)
    else:
        temp = xor('0' * pick, temp)
    
    return temp

def encode_data(data, generator):
    """Encode data with CRC."""
    l_gen = len(generator)
    appended_data = data + '0' * (l_gen - 1)
    remainder = mod2_division(appended_data, generator)
    return data + remainder

# Example usage
data = "110100111011"
generator = "1011"

print("Original Data:", data)
encoded_data = encode_data(data, generator)
print("Encoded Data:", encoded_data)
